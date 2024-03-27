using System.Net.Sockets;
using System.Net;

namespace TheST.Sockets
{
    public sealed class UdpMemorySender : IDisposable
    {
        private readonly Socket socket;
        private volatile bool disposed = false;
        private readonly IPEndPoint? remoteEndPoint;

        public UdpMemorySender(string remoteAddress, int remotePort)
        {
            socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
            remoteEndPoint = new IPEndPoint(IPAddress.Parse(remoteAddress), remotePort);
        }
        public UdpMemorySender()
        {
            socket = new Socket(AddressFamily.InterNetwork, SocketType.Dgram, ProtocolType.Udp);
        }

        public void Close()
        {
            socket.Close();
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Send(ReadOnlySpan<byte> data)
        {
            if(remoteEndPoint == null)
            {
                throw new InvalidOperationException("Remote endpoint is not set.");
            }
            Send(data, remoteEndPoint);
        }

        public void Send(ReadOnlySpan<byte> data, IPEndPoint to)
        {
            socket.SendTo(data, to);
        }
        public void Send(ReadOnlySpan<byte> data, string remoteHost, int remotePort)
        {
            Send(data, new IPEndPoint(IPAddress.Parse(remoteHost), remotePort));
        }

        private void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    // Dispose managed resources.
                    socket?.Close();
                    socket?.Dispose();
                }
                // Dispose unmanaged resources.
                disposed = true;
            }
        }

        ~UdpMemorySender()
        {
            Dispose(false);
        }
    }
}