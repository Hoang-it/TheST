import socket

HOST = "127.0.0.3"  # Standard loopback interface address (localhost)
PORT = 8888  # Port to listen on (non-privileged ports are > 1023)
package_size = 4096

class STSRealtimeAdapter:
    def __init__(self) -> None:
        self.host = HOST
        self.port = PORT
        self.package_size = package_size
        self.wave = []
        
    def open(self) -> socket:
        self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        host_address = (self.host, self.port)
        self.s.bind(host_address)
        print("Hosted on " + HOST + ":" + str(PORT))
        
        return self.s
    
    
    def listen(self):
        self.open()
        while True:
            # receive data
            data, addr = self.s.recvfrom(package_size)
            
            # inference
                       
            # send back
            self.send(data, ('127.0.0.1', 8888))  
                         
    def send(self, data, addr):
        self.s.sendto(data[:4096], addr)    
        
    def close(self) -> None:
        self.s.close()