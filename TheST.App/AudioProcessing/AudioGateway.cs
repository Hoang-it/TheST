using TheST.Sockets;

namespace TheST.App.AudioProcessing
{
    internal class AudioGateway : IAudioBufferDataHandler
    {
        private readonly IAudioBufferDataHandler _audioBufferDataHandler;

        private readonly IAudioPipeline _emptyPipeline;

        private readonly IAudioPipeline _pythonPipeline;

        private UdpCommunicator? _udpCommunicator;

        public AudioGateway(IAudioBufferDataHandler audioBufferDataHandler)
        {
            _pythonPipeline = new PythonProcessAudioPipeline(this);
            _emptyPipeline = new EmptyAudioPipeline(this);
            _audioBufferDataHandler = audioBufferDataHandler;
        }

        public bool ApplyAIEffect { get; set; } = false;

        //Receive from python
        public void ReceiveBuffer(ReadOnlySpan<byte> buffer)
        {
            _udpCommunicator?.SendBytes(buffer);
        }

        public void Send(ReadOnlySpan<byte> data)
        {
            SendToActivePipeline(data);
        }

        public void Start(string remoteAddress)
        {
            _udpCommunicator?.Dispose();
            _udpCommunicator = new UdpCommunicator(remoteAddress, 9999);
            _udpCommunicator.MessageReceived += UdpListener_MessageReceived;
            _udpCommunicator.StartListening();
            _pythonPipeline.Start();
            _emptyPipeline.Start();
        }

        public void Stop()
        {
            _pythonPipeline.Stop();
            _emptyPipeline.Stop();
        }

        private void SendToActivePipeline(ReadOnlySpan<byte> data)
        {
            if (ApplyAIEffect)
            {
                _pythonPipeline.Put(data);
                return;
            }
            _emptyPipeline.Put(data);
        }

        //Receive from partner
        private void UdpListener_MessageReceived(object? sender, ReadOnlyMemory<byte> e)
        {
            _audioBufferDataHandler?.ReceiveBuffer(e.Span);
        }
    }
}