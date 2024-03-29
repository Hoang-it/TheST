import socket
import numpy as np
from utilities.WaveUtilities import float_to_byte, byte_to_float, InputQueue
from model.trvc import TMA_RVC

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
HOST_PORT = 6666  # Port to listen on (non-privileged ports are > 1023)

CLIENT = '127.0.0.1'
CLIENT_PORT = 7777 

PACKAGE_SIZE = 32768
SAMPLERATE = 44100
CHANNELS = 1
BLOCK_TIME = 0.25
INQUEUE_MAXSIZE = 100_000

PTH_PATH = "assets\Indian-1-1000.pth"
INDEX_PATH = "assets\weights\\added_IVF49_Flat_nprobe_1_Indian-1-1000_v2.index"

class STSRealtimeAdapter:
    def __init__(self) -> None:
        self.host = HOST
        self.port = HOST_PORT
        self.package_size = PACKAGE_SIZE
        self.samplerate = SAMPLERATE
        self.channels = CHANNELS
        self.block_time = BLOCK_TIME        
        self.zc = self.samplerate // 100
        self.raw_block_frame = self.block_time * self.samplerate / self.zc
        self.block_frame = ( int(np.round(self.raw_block_frame)) * self.zc) 
        
        self.inqueue = InputQueue(maxsize=INQUEUE_MAXSIZE, blocksize=self.block_frame)
        self.model = TMA_RVC(samplerate=self.samplerate,
                             channels=self.channels,
                             block_time=self.block_time,
                             zc=self.zc,
                             raw_block_frame=self.raw_block_frame,
                             block_frame=self.block_frame,
                             pth_path=PTH_PATH,
                             index_path=INDEX_PATH)
        
    def open(self) -> socket:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host_address = (self.host, self.port)
        self.s.bind(host_address)
        print("Hosted on " + HOST + ":" + str(HOST_PORT))
        
        return self.s
    
    def decode(self, data: bytes) -> np.ndarray:
        return byte_to_float(data)
            
    def encode(self, data: np.ndarray) -> bytes:
        return float_to_byte(data)    
            
    def listen(self):
        """
            This is use for listen, process and sent data through UDP port.
        """
        self.open()
        try:
            while True:
                try:    
                    # receive data
                    print(f"======================================")
                    data, addr = self.s.recvfrom(PACKAGE_SIZE)  
                    print(f"input data size {len(data)}")   
                                                            
                    # convert data to numpy
                    decoded = self.decode(data=data)
                    self.inqueue.put(decoded)
                    
                    if self.inqueue.enough_data():
                        input_wave = self.inqueue.get()
                        print(f"input_wave {input_wave.shape}")
                        print(f"input_wave {input_wave}")

                        # inference
                        output_wave = self.model.infer(input_wave)  
                        print(f"output_wave {output_wave.shape}")
                        print(output_wave)
                        
                        # convert numpy to bytes                    
                        res = self.encode(output_wave)
                        
                        print(f"output datasize {len(res)}")
                                            
                        # send back
                        self.send(res, (CLIENT, CLIENT_PORT))  
                except KeyboardInterrupt:
                    print('Serivce shutdown')
                    break
                except Exception as e:
                    print(e)  
                    break
        finally:
            self.close()                                                                            
        
    def send(self, data, addr):
        self.s.sendto(data, addr)    
        
    def close(self) -> None:
        self.s.close()