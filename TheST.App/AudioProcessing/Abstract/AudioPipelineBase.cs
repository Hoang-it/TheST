namespace TheST.App.AudioProcessing
{
    internal abstract class AudioPipelineBase : IAudioPipeline
    {
        private readonly IAudioBufferDataHandler handler;
        private bool _disposed = false;

        public AudioPipelineBase(IAudioBufferDataHandler handler)
        {
            this.handler = handler;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public abstract void Put(ReadOnlySpan<byte> input);

        public abstract void Start();
        public abstract void Stop();

        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            if (disposing)
            {
                InternalDispose();
                _disposed = true;
            }
        }

        protected void HandleOutputData(ReadOnlySpan<byte> receivedBuffer)
        {
            handler.ReceiveBuffer(receivedBuffer);
        }

        protected abstract void InternalDispose();
    }
}