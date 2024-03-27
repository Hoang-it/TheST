namespace TheST.App.AudioProcessing
{
    internal sealed class EmptyAudioPipeline : AudioPipelineBase
    {
        private bool _isRunning;
        public EmptyAudioPipeline(IAudioBufferDataHandler handler) : base(handler)
        {
        }

        public override void Put(ReadOnlySpan<byte> input)
        {
            if (_isRunning == false)
            {
                return;
            }
            HandleOutputData(input);
        }

        public override void Start()
        {
            _isRunning = true;
        }
        public override void Stop()
        {
            _isRunning = false;
        }

        protected override void InternalDispose()
        {
            _isRunning = false;
        }
    }
}
