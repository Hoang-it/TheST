namespace TheST.App.AudioProcessing
{
    internal interface IAudioBufferDataHandler
    {
        void ReceiveBuffer(ReadOnlySpan<byte> buffer);
    }
}
