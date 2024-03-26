import time
import tools.rvc_for_realtime as rvc_for_realtime
from multiprocessing import Queue, cpu_count
from configs.config import Config
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from utilities.WaveUtilities import  printt
import torchaudio.transforms as tat
from tools.torchgate import TorchGate
import sys
import queue
import threading
        
class TMA_RVC:
    def __init__(self, 
                 samplerate, 
                 channels, 
                 block_time,
                 zc,
                 raw_block_frame,
                 block_frame) -> None:
        self.pitch: int = 0
        self.pth_path: str = "F:\.Net\TheST\PythonServer\\assets\Indian-1-1000.pth"
        self.index_path: str = "F:\.Net\TheST\PythonServer\\assets\weights\\added_IVF49_Flat_nprobe_1_Indian-1-1000_v2.index"
        self.index_rate: float = 0.1
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
        self.block_time = block_time
        self.rms_mix_rate = 0.5
        # self.samplerate = self.get_device_samplerate()
        self.samplerate = samplerate
        self.zc = zc
        # self.channels = self.get_device_channels()
        self.channels = channels
        self.raw_block_frame = raw_block_frame
        self.block_frame = block_frame
        self.device = torch.device('cuda')

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
        self.resampler = tat.Resample(
                orig_freq=self.samplerate,
                new_freq=16000,
                dtype=torch.float32,
            ).to(self.device)
        
        self.input_wav_res: torch.Tensor = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.device,
            dtype=torch.float32,
        )            
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.skip_head = self.extra_frame // self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.return_length = (
                        self.block_frame + self.sola_buffer_frame + self.sola_search_frame
                    ) // self.zc
        

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
        self.threhold = -60
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.I_noise_reduce: bool = False
        self.input_wav_denoise: torch.Tensor = self.input_wav.clone()
        self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")
        self.function = "vc"
        self.tg = TorchGate(
                sr=self.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
            ).to(self.device)
        self.nr_buffer: torch.Tensor = self.sola_buffer.clone()
        # self.resampler2 = None
        if self.rvc.tgt_sr != self.samplerate:
                self.resampler2 = tat.Resample(
                    orig_freq=self.rvc.tgt_sr,
                    new_freq=self.samplerate,
                    dtype=torch.float32,
                ).to(self.device)
        else:
            self.resampler2 = None
        self.O_noise_reduce: bool = False
        self.output_buffer: torch.Tensor = self.input_wav.clone()
        self.use_pv: bool = False
        
    def infer(self, indata: np.ndarray):
        print(f"================={indata.shape}=======================")
        start_time = time.perf_counter()
        indata = librosa.to_mono(indata.T)
        if self.threhold > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < self.threhold
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        
        print(f"Line 170: indata {indata.shape}, block_frame {self.block_frame}")
        print(f"input_wav size {self.input_wav.shape}")
        self.input_wav[: -self.block_frame] = self.input_wav[
            self.block_frame :
        ].clone()
        print(f"Line 175 input_wav: {self.input_wav.shape}")
        
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(
            self.device
        )
        print(f"Line 180 input_wav: {self.input_wav.shape}")
        
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        print(f"Line 185 input_wav: {self.input_wav.shape}")
        
        # input noise reduction and resampling
        if self.I_noise_reduce:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += (
                self.nr_buffer * self.fade_out_window
            )
            self.input_wav_denoise[-self.block_frame :] = input_wav[
                : self.block_frame
            ]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(
                self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            )[160:]
        else:
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (
                self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[
                    160:
                ]
            )
        # infer
        print(f"Line 214 input_wav: {self.input_wav.shape}")
        
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.f0method,
            )
            print(f"Line 224: {infer_wav.shape}")
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
        elif self.I_noise_reduce:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        print(f"Line 231: {infer_wav.shape}")
        
        # output noise reduction
        if self.O_noise_reduce and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        print(f"Line 242: {infer_wav.shape}")
        
        # volume envelop mixing
        if self.rms_mix_rate < 1 and self.function == "vc":
            if self.I_noise_reduce:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(
                rms1 / rms2, torch.tensor(1 - self.rms_mix_rate)
            )
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        print(f"Line 279: {infer_wav.shape}")
        
        conv_input = infer_wav[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input**2,
                torch.ones(1, 1, self.sola_buffer_frame, device=self.device),
            )
            + 1e-8
        )
        print(f"Line 291: {infer_wav.shape}")
        
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        printt("sola_offset = %d", int(sola_offset))
        print(f"Line 300: {infer_wav.shape}")
        
        infer_wav = infer_wav[sola_offset:]
        print(f"Line 303: {infer_wav.shape}")
        
        if "privateuseone" in str(self.config.device) or not self.use_pv:
            infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
            infer_wav[: self.sola_buffer_frame] += (
                self.sola_buffer * self.fade_out_window
            )
        else:
            infer_wav[: self.sola_buffer_frame] = self.phase_vocoder(
                self.sola_buffer,
                infer_wav[: self.sola_buffer_frame],
                self.fade_out_window,
                self.fade_in_window,
            )
        print(f"Line 317: {infer_wav.shape}")
        
        print("Debug")
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        print(f"Line 323: {infer_wav.shape}")
        
        print("Debug")
        
        total_time = time.perf_counter() - start_time
        printt("Infer time: %.2f", total_time)
        return (
                infer_wav[: self.block_frame]
                # .repeat(self.channels, 1)
                .repeat(self.channels, 1)
                .t()
                .cpu()
                .numpy()
            )                        
        
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
    
    def audio_callback(
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status
        ):           
            outdata[:] = self.infer(indata=indata)
            