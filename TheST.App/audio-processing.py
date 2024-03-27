import socket

HOST = "127.0.0.1"  # Standard loopback interface address (localhost)
PORT = 6666  # Port to listen on (non-privileged ports are > 1023)
package_size = 32768

class STSRealtimeAdapter:
    def __init__(self) -> None:
        self.host = HOST
        self.port = PORT
        self.package_size = package_size
        self.samplerate = 44100
        self.channels = 1
        self.block_time = 0.25                
        
    def open(self) -> socket:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host_address = (self.host, self.port)
        self.s.bind(host_address)
        print("Hosted on " + HOST + ":" + str(PORT))
        
        return self.s
    
    def listen(self):
        try:
            self.open()
            while True:
                try:    
                    # receive data
                    data, addr = self.s.recvfrom(package_size)  
                    print(f"input data size {len(data)}")   
                                                            
                    # convert numpy to bytes    

                    # send back
                    self.send(data, ('127.0.0.1', 7777))  
                except KeyboardInterrupt:
                    print('\nRecording finished')
                    break
                except Exception as e:
                    print(e)  
                    break
        except KeyboardInterrupt:
            print('\nRecording finished')
        finally:
            self.close()                                                                        
        
    def send(self, data, addr):
        self.s.sendto(data, addr)    
        
    def close(self) -> None:
        self.s.close()
    
if __name__ == "__main__":
    try:
        adapter = STSRealtimeAdapter()
        adapter.listen()
    except KeyboardInterrupt:
        print('\nRecording finished')
    except Exception as e:
        print(e)