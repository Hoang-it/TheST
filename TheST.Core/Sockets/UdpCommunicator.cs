using System.Net.Sockets;
using System.Net;
using TheST.Core.Buffers;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace TheST.Sockets
{
    public sealed class UdpCommunicator : IDisposable
    {
        private readonly UdpClient _udpClient;
        private bool isListening;
        private Task? receiveTask;
        private IPEndPoint remoteEndPoint;

        public UdpCommunicator(string remoteAddress, int remotePort)
        {
            _udpClient = new UdpClient();
            remoteEndPoint = new IPEndPoint(IPAddress.Parse(remoteAddress), remotePort);
            isListening = false;
        }

        public event EventHandler<ReadOnlyMemory<byte>>? MessageReceived;
        public event EventHandler<string>? OnError;
        public void Dispose()
        {
            try
            {
                MessageReceived = null;
                OnError = null;
                _udpClient?.Close();
                _udpClient?.Dispose();
            }
            catch (Exception)
            {
            }
        }

        public void SendBytes(ReadOnlySpan<byte> bytes)
        {
            using (var tempBuffer = new RentedMemory<byte>(bytes))
            {
                _udpClient.Send(tempBuffer, tempBuffer.Length, remoteEndPoint);
            }
        }

        public void SendBytes(ReadOnlySpan<byte> bytes, string remoteAddress, int remotePort)
        {
            using (var tempBuffer = new RentedMemory<byte>(bytes))
            {
                _udpClient.Send(tempBuffer, tempBuffer.Length, new IPEndPoint(IPAddress.Parse(remoteAddress), remotePort));
            }
        }

        public void StartListening()
        {
            if (isListening)
            {
                return;
            }
            receiveTask = Task.Run(ListenForMessages);
        }

        private void ListenForMessages()
        {
            if (isListening)
            {
                return;
            }
            isListening = true;
            _udpClient.Client.Bind(new IPEndPoint(IPAddress.Any, remoteEndPoint.Port));
            while (isListening)
            {
                try
                {
                    IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Any, 0);
                    byte[] data = _udpClient.Receive(ref localEndPoint);
                    MessageReceived?.Invoke(this, data);
                }
                catch (SocketException) { }
                catch (Exception ex)
                {
                    OnError?.Invoke(this, ex.Message);
                }
            }
        }
    }
}