import sounddevice as sd
import numpy as np
import argparse
import queue
import sys
import threading
import librosa
import torch
import torch.nn.functional as F

def get_device_channels():
    max_input_channels = sd.query_devices(device=sd.default.device[0])[
        "max_input_channels"
    ]
    max_output_channels = sd.query_devices(device=sd.default.device[1])[
        "max_output_channels"
    ]
    return min(max_input_channels, max_output_channels, 2)
        
def get_device_samplerate():
    return int(
        sd.query_devices(device=sd.default.device[0])["default_samplerate"]
    )

def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result
    
duration = 5.5  # seconds
block_time = 0.25
samplerate = get_device_samplerate()
zc = samplerate // 100
channels = get_device_channels()
raw_block_frame = block_time * samplerate / zc
block_frame = ( int(np.round(raw_block_frame)) * zc)
device = torch.device('cpu')

extra_time: float = 2.5
extra_frame = (
    int(
        np.round(extra_time * samplerate / zc)
    ) * zc
)

crossfade_time: float = 0.05
crossfade_frame = (
    int(
        np.round(crossfade_time * samplerate / zc)
    ) * zc
)

sola_search_frame = zc

input_wav: torch.Tensor = torch.zeros(
    extra_frame
    + crossfade_frame
    + sola_search_frame
    + block_frame,
    device=device,
    dtype=torch.float32,
)

input_wav_res: torch.Tensor = torch.zeros(
    160 * input_wav.shape[0] // zc,
    device=device,
    dtype=torch.float32,
)            
block_frame_16k = 160 * block_frame // zc

sola_buffer_frame = min(crossfade_frame, 4 * zc)

sola_buffer: torch.Tensor = torch.zeros(
    sola_buffer_frame, 
    device=device, 
    dtype=torch.float32
)

fade_in_window: torch.Tensor = (
    torch.sin(
        0.5 * np.pi * torch.linspace(
            0.0, 1.0,steps=sola_buffer_frame,
            device=device,
            dtype=torch.float32,
        )
    )** 2
)
fade_out_window: torch.Tensor = 1 - fade_in_window
f0method: str = "fcpe"

buffersize = 20
q = queue.Queue(maxsize=buffersize)


if __name__ == "__main__":
    print("Loading")
    # Load model
    import tools.rvc_for_realtime as rvc_for_realtime
    from multiprocessing import Queue, cpu_count, freeze_support
    from configs.config import Config
    freeze_support()    

    pitch: int = 0
    pth_path: str = "F:\.Net\TheST\PythonServer\checkpoints\\357k.pth"
    index_path: str = ""
    index_rate: float = 0.0
    n_cpu = min(cpu_count(), 8)
    n_cpu: int = min(n_cpu, 4)
    inp_q = Queue()
    opt_q = Queue()
    config = Config()

    rvc = rvc_for_realtime.RVC(
        pitch,
        pth_path,
        index_path,
        index_rate,
        n_cpu,
        inp_q,
        opt_q,
        config,
        None,
    )
    
    def callback(indata, outdata, frames, time, status):
        # print(type(indata))
        print(indata.shape)
        if status:
            print(status)
        
        # convert to mono sound
        indata = librosa.to_mono(indata.T)
        
        # copy last block to head of input wav
        input_wav[: -block_frame] = input_wav[block_frame: ].clone()
        
        # copy indata to last
        indatasize = indata.shape[0]
        input_wav[-indatasize :] = torch.from_numpy(indata).to(device)
        
        # copy last block to head of response wav 
        input_wav_res[: -block_frame_16k] = input_wav_res[block_frame_16k :].clone()
        
        # infer
        infer_wav = input_wav[extra_frame :].clone()
        # try: 
        #     skip_head = extra_frame // zc
        #     return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc    
            
        #     infer_wav = rvc.infer(
        #         input_wav_res,
        #         block_frame_16k,
        #         skip_head,
        #         return_length,
        #         f0method,
        #     )
        # except Exception as e:
        #     print(e)
        #     infer_wav = input_wav[extra_frame :].clone()
                    
        # calc sola_offset
        conv_input = infer_wav[None, None, : sola_buffer_frame + sola_search_frame]
        
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, sola_buffer_frame, device=device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        print(f"sola_offset {int(sola_offset)}")
        
        # take sola_offset
        infer_wav = infer_wav[sola_offset:]
        
        # decode phase
        infer_wav[: sola_buffer_frame] = phase_vocoder(
                        sola_buffer,
                        infer_wav[: sola_buffer_frame],
                        fade_out_window,
                        fade_in_window,
                    )
        
        # response
        
        outdata[:] = (
            infer_wav[: block_frame]
            .repeat(channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        print(indata.shape)
        print(outdata.shape)
        print(samplerate)
        print(channels)
        q.put(indata.copy())

    print("Model loaded")
    
    stream = sd.Stream(
            callback=callback,
            blocksize=block_frame,
            samplerate=samplerate,
            channels=channels,
            dtype="float32",
            extra_settings=None,
        )
    stream.start()
    try:    
        print('press Ctrl+C to stop the recording')
        while True:
            try:
                data = q.get()
                # print(data)
            except KeyboardInterrupt:
                print('\nRecording finished')
                break
    except KeyboardInterrupt:
        print('\nRecording finished')