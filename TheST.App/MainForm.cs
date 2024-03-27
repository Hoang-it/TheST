using Audio;
using Audio.Capture;
using Audio.Playback;
using NAudio.Wave;
using System.Net;
using System.Windows.Forms;
using TheST.App.AudioProcessing;
using TheST.App.Controls;
using TheST.App.EventArguments;
using TheST.Models;

namespace TheST.App
{
    public partial class MainForm : Form, IAudioBufferDataHandler
    {
        const string StartLabel = "Start";
        const string StopLabel = "Stop";
        private readonly IAudioCapture _audioCapture;
        private readonly IAudioPlayback _audioPlayback;
        private WaveFormat _waveFormat = new WaveFormat(44100, 16, 1);

        private readonly AudioGateway _audioGateway;
        public MainForm()
        {
            InitializeComponent();
            _audioCapture = new AudioCapture(_waveFormat);
            _audioCapture.DataAvailable += AudioCapture_DataAvailable;
            _audioPlayback = new AudioPlayback(_waveFormat);
            _startButton.Text = StartLabel;
            _audioGateway = new AudioGateway(this);
        }

        protected override void OnLoad(EventArgs e)
        {
            base.OnLoad(e);
            _waveFormatInfo.Title = "Playback Wave Format";
            _waveFormatInfo.WaveFormat = _audioCapture.WaveFormat;
            _waveFormatConfiguration.SelectedWaveFormatChanged += WaveFormatChanged;
        }

        private void WaveFormatChanged(object? sender, WaveFormat? e)
        {
            if (e == null)
            {
                return;
            }

            //_audioCapture.WaveFormat = e;
            //_audioPlayback.WaveFormat = e;
            //_waveFormatInfo.WaveFormat = e;
        }

        private void AudioCapture_DataAvailable(object? sender, ReadOnlyMemory<byte> inputSamples)
        {
            _audioGateway.Send(inputSamples.Span);
        }
        private bool IsValidIpAddress(string ipAddress)
        {
            return IPAddress.TryParse(ipAddress, out _);
        }
        private void HandleStartButtonClick(object sender, EventArgs e)
        {
            if (_startButton.Text == StartLabel)
            {
                var remoteAddress = _txtRemoteAddress.Text;
                if (!IsValidIpAddress(remoteAddress))
                {
                    _ipAddressInvalidError.SetError(_txtRemoteAddress, "Please enter a valid IP address");
                    return;
                }
                _ipAddressInvalidError.Clear();
                _audioCapture.StartCapturing();
                _audioPlayback.Play();
                _audioGateway.Start(remoteAddress);
                _startButton.Text = StopLabel;
                _waveFormatConfiguration.Enabled = false;
                _deviceConfiguration.Enabled = false;
                _txtRemoteAddress.Enabled = false;
            }
            else
            {
                _startButton.Enabled = true;
                _audioCapture.StopCapturing();
                _audioPlayback.Stop();
                _audioGateway.Stop();
                _startButton.Text = StartLabel;
                _waveFormatConfiguration.Enabled = true;
                _deviceConfiguration.Enabled = true;
                _txtRemoteAddress.Enabled = true;
            }
        }

        private void CaptureDeviceChanged(object sender, DeviceChangedArgs e)
        {
            if (sender is DeviceConfiguration deviceConfiguration &&
                deviceConfiguration.SelectedCaptureDevice is Device selectedDevice)
            {
                _audioCapture.UpdateCaptureDevice(selectedDevice.Id);
            }
        }

        private void PlaybackDeviceChanged(object sender, DeviceChangedArgs e)
        {
            if (sender is DeviceConfiguration deviceConfiguration &&
                deviceConfiguration.SelectedPlaybackDevice is Device selectedDevice)
            {
                _audioPlayback.UpdatePlaybackDevice(selectedDevice.Id);
            }
        }

        public void ReceiveBuffer(ReadOnlySpan<byte> buffer)
        {
            _audioPlayback.AddSample(buffer);
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            _audioGateway.ApplyAIEffect = checkBox1.Checked;
        }
    }
}