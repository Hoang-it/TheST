import socket
from multiprocessing import freeze_support
import numpy as np
from scipy.io.wavfile import write
from utilities.WaveUtilities import float_to_byte, byte_to_float, printt
import sounddevice as sd
import sys
import queue
import threading
from model.trvc import TMA_RVC

HOST = "127.0.0.3"  # Standard loopback interface address (localhost)
PORT = 8888  # Port to listen on (non-privileged ports are > 1023)
package_size = 32768
            
class STSRealtimeAdapter:
    def __init__(self) -> None:
        self.host = HOST
        self.port = PORT
        self.package_size = package_size
        self.samplerate = 44100
        self.channels = 1
        self.block_time = 0.25        
        self.zc = self.samplerate // 100
        self.raw_block_frame = self.block_time * self.samplerate / self.zc
        self.block_frame = ( int(np.round(self.raw_block_frame)) * self.zc)   
        print(f"block_frame {self.block_frame}")     
        
        self.model = TMA_RVC(samplerate=self.samplerate,
                             channels=self.channels,
                             block_time=self.block_time,
                             zc=self.zc,
                             raw_block_frame=self.raw_block_frame,
                             block_frame=self.block_frame)
        
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
        """
        This is use for take data sent from UDP port
        """
        self.open()
        while True:
            try:    
                # receive data
                print("======================================")
                data, addr = self.s.recvfrom(package_size)  
                print(f"input data size {len(data)}")   
                                           
                # convert data to numpy
                input_wave = self.decode(data=data)
                                
                # inference
                output_wave = self.model.infer(input_wave)  
                print(f"output_wave {output_wave.shape}")
                print(output_wave)
                
                # convert numpy to bytes
                res = self.encode(output_wave)
                print(f"output datasize {len(res)}")
                                    
                # send back
                self.send(res, ('127.0.0.1', 8888))  
            except KeyboardInterrupt:
                print('\nRecording finished')
                break
            except Exception as e:
                print(e)   
            finally:
                self.close()                         
                          
    def sounddevice(self):
        """
        This is use for test with sounddevice, like in repo
        """
        try:    
            stream = sd.Stream(
                callback=self.model.audio_callback,
                blocksize=self.block_frame,
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="float32",
                extra_settings=None,
            )
            stream.start()
            print('press Ctrl+C to stop the converting')
            while True:
                try:
                    print('.')
                    # continue
                except KeyboardInterrupt:
                    print('\Converting finished')
                    break
        except KeyboardInterrupt:
            print('\Converting finished')
        
    def send(self, data, addr):
        self.s.sendto(data, addr)    
        
    def close(self) -> None:
        self.s.close()