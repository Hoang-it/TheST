namespace TheST.App.AudioProcessing
{
    internal interface IAudioPipeline : IDisposable
    {
        void Put(ReadOnlySpan<byte> input);
        void Start();
        void Stop();
    }
}
