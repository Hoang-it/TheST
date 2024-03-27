using Audio.Devices;
using Audio.WaveProviders;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace Audio.Playback
{
    public sealed class AudioPlayback : IAudioPlayback
    {
        private BufferedWaveProviderExtend? _bufferedWaveProvider;
        private WaveFormat _currentWaveFormat;
        private string? _deviceId;
        private WasapiOut? _waveOut;
        public WaveFormat WaveFormat
        {
            get => _currentWaveFormat;
            set
            {
                _currentWaveFormat = value;
                UpdatePlaybackDeviceInternal(_deviceId ?? string.Empty);
            }
        }
        public bool IsPlaying => _waveOut?.PlaybackState == PlaybackState.Playing;

        public AudioPlayback(WaveFormat waveFormat)
        {
            _currentWaveFormat = waveFormat;
            UpdatePlaybackDeviceInternal(string.Empty);
        }

        public void AddSample(ReadOnlySpan<byte> sample)
        {
            _bufferedWaveProvider?.AddSamples(sample);  
        }

        public void Pause()
        {
            _waveOut?.Pause();
        }

        public void Play()
        {
            _bufferedWaveProvider?.ClearBuffer();
            _waveOut?.Play();
        }

        public void Stop()
        {
            _waveOut?.Stop();
        }

        public bool UpdatePlaybackDevice(string deviceId)
        {
            return UpdatePlaybackDeviceInternal(deviceId);
        }

        private bool UpdatePlaybackDeviceInternal(string deviceId)
        {
            try
            {
                if (!DevicesFactory.PlaybackDeviceManager.TryGetDevice(deviceId, out var device) && device is null)
                {
                    return false;
                }
                _deviceId = deviceId;
                var currentPlaybackState = IsPlaying;
                if (_waveOut != null)
                {
                    _waveOut.Stop();
                    _waveOut.Dispose();
                }

                _waveOut = new WasapiOut(device, AudioClientShareMode.Shared, false, 100);
                _bufferedWaveProvider = new BufferedWaveProviderExtend(_currentWaveFormat);
                _waveOut.Init(_bufferedWaveProvider);
                if (currentPlaybackState)
                {
                    _waveOut.Play();
                }
                return true;
            }
            catch
            {
                return false;
            }
        }
    }
}