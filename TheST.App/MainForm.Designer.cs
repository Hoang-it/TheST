using NAudio.Wave;

namespace TheST.App
{
    partial class MainForm
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            _deviceConfiguration = new Controls.DeviceConfiguration();
            _waveFormatInfo = new Controls.WaveFormatInfo();
            _startButton = new Button();
            _waveFormatConfiguration = new Controls.WaveFormatConfiguration();
            SuspendLayout();
            // 
            // _deviceConfiguration
            // 
            _deviceConfiguration.Location = new Point(14, 16);
            _deviceConfiguration.Margin = new Padding(3, 5, 3, 5);
            _deviceConfiguration.MaximumSize = new Size(366, 160);
            _deviceConfiguration.MinimumSize = new Size(366, 160);
            _deviceConfiguration.Name = "_deviceConfiguration";
            _deviceConfiguration.Size = new Size(366, 160);
            _deviceConfiguration.TabIndex = 0;
            _deviceConfiguration.CaptureDeviceChanged += CaptureDeviceChanged;
            _deviceConfiguration.PlaybackDeviceChanged += PlaybackDeviceChanged;
            // 
            // _waveFormatInfo
            // 
            _waveFormatInfo.Location = new Point(745, 16);
            _waveFormatInfo.Margin = new Padding(3, 5, 3, 5);
            _waveFormatInfo.MaximumSize = new Size(263, 160);
            _waveFormatInfo.MinimumSize = new Size(263, 160);
            _waveFormatInfo.Name = "_waveFormatInfo";
            _waveFormatInfo.Size = new Size(263, 160);
            _waveFormatInfo.TabIndex = 1;
            _waveFormatInfo.Title = "Wave format";
            _waveFormatInfo.WaveFormat = null;
            // 
            // _startButton
            // 
            _startButton.Location = new Point(386, 145);
            _startButton.Margin = new Padding(3, 4, 3, 4);
            _startButton.Name = "_startButton";
            _startButton.Size = new Size(352, 31);
            _startButton.TabIndex = 2;
            _startButton.Text = "Start capture";
            _startButton.UseVisualStyleBackColor = true;
            _startButton.Click += HandleStartButtonClick;
            // 
            // _waveFormatConfiguration
            // 
            _waveFormatConfiguration.Location = new Point(386, 16);
            _waveFormatConfiguration.Margin = new Padding(3, 5, 3, 5);
            _waveFormatConfiguration.MaximumSize = new Size(352, 83);
            _waveFormatConfiguration.MinimumSize = new Size(352, 83);
            _waveFormatConfiguration.Name = "_waveFormatConfiguration";
            _waveFormatConfiguration.Size = new Size(352, 83);
            _waveFormatConfiguration.TabIndex = 5;
            _waveFormatConfiguration.Title = "Wave format";
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(1016, 509);
            Controls.Add(_waveFormatConfiguration);
            Controls.Add(_startButton);
            Controls.Add(_waveFormatInfo);
            Controls.Add(_deviceConfiguration);
            Margin = new Padding(3, 4, 3, 4);
            Name = "MainForm";
            Text = "Form1";
            FormClosing += MainForm_FormClosing;
            ResumeLayout(false);
        }

        #endregion

        private Controls.DeviceConfiguration _deviceConfiguration;
        private Controls.WaveFormatInfo _waveFormatInfo;
        private Button _startButton;
        private Controls.WaveFormatConfiguration _waveFormatConfiguration;
    }
}
