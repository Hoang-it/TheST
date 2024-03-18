using Audio;
using Audio.Capture;
using Audio.Playback;
using NAudio.Wave;
using System.Runtime.InteropServices;
using TheST.App.Controls;
using TheST.App.EventArguments;
using TheST.Core.Buffers;
using TheST.Models;
using TheST.Sockets;

namespace TheST.App
{
    public partial class MainForm : Form
    {
        const string StartLabel = "Start";
        const string StopLabel = "Stop";
        private readonly IAudioCapture _audioCapture;
        private readonly IAudioPlayback _audioPlayback;
        private WaveFormat _waveFormat = new WaveFormat(8000, 16, 1);
        private readonly UdpMemorySender _udpSender;
        private UdpCommunicator _udpListener;
        public MainForm()
        {
            InitializeComponent();
            _audioCapture = new AudioCapture(_waveFormat);
            _audioCapture.DataAvailable += AudioCapture_DataAvailable;
            _audioPlayback = new AudioPlayback(_waveFormat);
            _startButton.Text = StartLabel;
            _udpSender = new UdpMemorySender();
            _udpListener = new UdpCommunicator("127.0.0.1", 8888); // TODO: Move to UI configuration
            _udpListener.MessageReceived += _udpListener_MessageReceived;
            _udpListener.StartListening();
        }

        private void _udpListener_MessageReceived(object? sender, ReadOnlyMemory<byte> receivedBuffer)
        {
            _audioPlayback.AddSample(receivedBuffer.Span);
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

            _audioCapture.WaveFormat = e;
            _audioPlayback.WaveFormat = e;
            _waveFormatInfo.WaveFormat = e;
        }

        private void AudioCapture_DataAvailable(object? sender, ReadOnlyMemory<byte> inputSamples)
        {
            _udpSender.Send(inputSamples, "127.0.0.3", 8888); // TODO: Move to UI configuration
        }

        private void HandleStartButtonClick(object sender, EventArgs e)
        {
            if (_startButton.Text == StartLabel)
            {
                _audioCapture.StartCapturing();
                _audioPlayback.Play();
                _startButton.Text = StopLabel;
                _waveFormatConfiguration.Enabled = false;
                _deviceConfiguration.Enabled = false;
            }
            else
            {
                _startButton.Enabled = true;
                _audioCapture.StopCapturing();
                _audioPlayback.Stop();
                _startButton.Text = StartLabel;
                _waveFormatConfiguration.Enabled = true;
                _deviceConfiguration.Enabled = true;
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

        private void MainForm_FormClosing(object sender, FormClosingEventArgs e)
        {

        }
    }
}