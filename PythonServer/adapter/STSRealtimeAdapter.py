import socket
import tools.rvc_for_realtime as rvc_for_realtime
from multiprocessing import Queue, cpu_count, freeze_support
from configs.config import Config
import numpy as np
import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import io
from miniaudio import SampleFormat, decode, wav_read_f32
from scipy.io.wavfile import write
import sounddevice as sd
from utilities.WaveUtilities import float_to_byte, byte_to_float
HOST = "127.0.0.3"  # Standard loopback interface address (localhost)
PORT = 8888  # Port to listen on (non-privileged ports are > 1023)
package_size = 32768

class TMA_RVC:
    def __init__(self, samplerate, channels) -> None:
        self.pitch: int = 0
        self.pth_path: str = "G:\RVC1006AMD_Intel\RVC1006AMD_Intel1\\assets\weights\\kikiV1.pth"
        self.index_path: str = ""
        self.index_rate: float = 0.0
        self.n_cpu = min(cpu_count(), 8)
        self.n_cpu: int = min(self.n_cpu, 4)
        self.inp_q = Queue()
        self.opt_q = Queue()
        self.config = Config()
        self.rvc = rvc_for_realtime.RVC(
            self.pitch,
            self.pth_path,
            self.index_path,
            self.index_rate,
            self.n_cpu,
            self.inp_q,
            self.opt_q,
            self.config,
            None,
        )
        
        self.duration = 5.5  # seconds
        self.block_time = 0.25
        # self.samplerate = self.get_device_samplerate()
        self.samplerate = samplerate
        self.zc = self.samplerate // 100
        # self.channels = self.get_device_channels()
        self.channels = channels
        self.raw_block_frame = self.block_time * self.samplerate / self.zc
        self.block_frame = ( int(np.round(self.raw_block_frame)) * self.zc)
        self.device = torch.device('cpu')

        self.extra_time: float = 2.5
        self.extra_frame = (
            int(
                np.round(self.extra_time * self.samplerate / self.zc)
            ) * self.zc
        )

        self.crossfade_time: float = 0.05
        self.crossfade_frame = (
            int(
                np.round(self.crossfade_time * self.samplerate / self.zc)
            ) * self.zc
        )

        self.sola_search_frame = self.zc

        self.input_wav: torch.Tensor = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.device,
            dtype=torch.float32,
        )

        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.device,
            dtype=torch.float32,
        )            
        self.block_frame_16k = 160 * self.block_frame // self.zc

        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)

        self.sola_buffer: torch.Tensor = torch.zeros(
            self.sola_buffer_frame, 
            device=self.device, 
            dtype=torch.float32
        )

        self.fade_in_window: torch.Tensor = (
            torch.sin(
                0.5 * np.pi * torch.linspace(
                    0.0, 1.0,steps=self.sola_buffer_frame,
                    device=self.device,
                    dtype=torch.float32,
                )
            )** 2
        )
        self.fade_out_window: torch.Tensor = 1 - self.fade_in_window
        self.f0method: str = "fcpe"
    
    def inference(self, indata: np.ndarray) -> np.ndarray:
        # convert to mono sound
        indata = librosa.to_mono(indata.T)
        
        # copy last block to head of input wav
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame: ].clone()
        
        # copy indata to last
        indatasize = indata.shape[0]
        self.input_wav[-indatasize :] = torch.from_numpy(indata).to(self.device)
        
        # copy last block to head of response wav 
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        
        # infer
        # infer_wav = self.input_wav[self.extra_frame :].clone()
        try: 
            skip_head = self.extra_frame // self.zc
            return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc    
            
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                skip_head,
                return_length,
                self.f0method,
            )
        except Exception as e:
            print(e)
            infer_wav = self.input_wav[self.extra_frame :].clone()
                    
        # calc sola_offset
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8
        )
        sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        print(f"sola_offset {int(sola_offset)}")
        
        # take sola_offset
        infer_wav = infer_wav[sola_offset:]
        
        # decode phase
        infer_wav[: self.sola_buffer_frame] = self.phase_vocoder(
                        self.sola_buffer,
                        infer_wav[: self.sola_buffer_frame],
                        self.fade_out_window,
                        self.fade_in_window,
                    )
        
        # response
        
        outdata = (
            infer_wav[: self.block_frame]
            .repeat(self.channels, 1)
            .t()
            .cpu()
            .numpy()
        )
        
        return outdata    

    def phase_vocoder(self, a, b, fade_out, fade_in):
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
    
class STSRealtimeAdapter:
    def __init__(self) -> None:
        self.host = HOST
        self.port = PORT
        self.package_size = package_size
        self.samplerate = 44100
        self.channels = 2
        self.model = TMA_RVC(samplerate=self.samplerate,
                             channels=self.channels)
        
    def open(self) -> socket:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host_address = (self.host, self.port)
        self.s.bind(host_address)
        print("Hosted on " + HOST + ":" + str(PORT))
        
        return self.s
    
    def decode(self, data: bytes) -> np.ndarray:
        return byte_to_float(data)
            
    def encode(self, data: np.ndarray) -> bytes:
        return float_to_byte(data)
        
    def listen(self):
        self.open()
        data_size = 10
        while True:
            try:    
                # receive data
                data, addr = self.s.recvfrom(package_size)                                
                # convert data to numpy
                input_wave = self.decode(data=data)                                    
                
                # inference
                # output_wave = self.model.inference(input_wave)  
                
                # convert numpy to bytes
                res = self.encode(input_wave)
                print(len(res))
                                    
                # send back
                self.send(res, ('127.0.0.1', 8888))  
            except KeyboardInterrupt:
                print('\nRecording finished')
                break
            except Exception as e:
                print(e)            
            
                         
    def send(self, data, addr):
        self.s.sendto(data, addr)    
        
    def close(self) -> None:
        self.s.close()