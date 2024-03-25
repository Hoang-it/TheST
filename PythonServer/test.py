import tools.rvc_for_realtime as rvc_for_realtime
from multiprocessing import Queue, cpu_count, freeze_support
from configs.config import Config
    
def main():  
    """
    This is just use for testing feature of python and related libs
    """  
    pitch: int = 0
    pth_path: str = "F:\.Net\TheST\PythonServer\\assets\Indian-1-1000.pth"
    index_path: str = "F:\.Net\TheST\PythonServer\\assets\weights\\added_IVF49_Flat_nprobe_1_Indian-1-1000_v2.index"
    index_rate: float = 0.0
    n_cpu = min(cpu_count(), 8)
    n_cpu: int = min(n_cpu, 4)
    inp_q = Queue()
    opt_q = Queue()
    f0method: str = "fcpe"
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
    
if __name__ == '__main__':
    freeze_support()    
    main()