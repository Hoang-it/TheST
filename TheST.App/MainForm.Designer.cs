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
            components = new System.ComponentModel.Container();
            _deviceConfiguration = new Controls.DeviceConfiguration();
            _waveFormatInfo = new Controls.WaveFormatInfo();
            _startButton = new Button();
            _waveFormatConfiguration = new Controls.WaveFormatConfiguration();
            groupBox1 = new GroupBox();
            _txtRemoteAddress = new TextBox();
            label1 = new Label();
            _ipAddressInvalidError = new ErrorProvider(components);
            checkBox1 = new CheckBox();
            groupBox1.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)_ipAddressInvalidError).BeginInit();
            SuspendLayout();
            // 
            // _deviceConfiguration
            // 
            _deviceConfiguration.Location = new Point(12, 12);
            _deviceConfiguration.MaximumSize = new Size(320, 120);
            _deviceConfiguration.MinimumSize = new Size(320, 120);
            _deviceConfiguration.Name = "_deviceConfiguration";
            _deviceConfiguration.Size = new Size(320, 120);
            _deviceConfiguration.TabIndex = 0;
            _deviceConfiguration.CaptureDeviceChanged += CaptureDeviceChanged;
            _deviceConfiguration.PlaybackDeviceChanged += PlaybackDeviceChanged;
            // 
            // _waveFormatInfo
            // 
            _waveFormatInfo.Location = new Point(652, 12);
            _waveFormatInfo.MaximumSize = new Size(230, 120);
            _waveFormatInfo.MinimumSize = new Size(230, 120);
            _waveFormatInfo.Name = "_waveFormatInfo";
            _waveFormatInfo.Size = new Size(230, 120);
            _waveFormatInfo.TabIndex = 1;
            _waveFormatInfo.Title = "Wave format";
            _waveFormatInfo.WaveFormat = null;
            // 
            // _startButton
            // 
            _startButton.Location = new Point(338, 109);
            _startButton.Name = "_startButton";
            _startButton.Size = new Size(308, 23);
            _startButton.TabIndex = 2;
            _startButton.Text = "Start capture";
            _startButton.UseVisualStyleBackColor = true;
            _startButton.Click += HandleStartButtonClick;
            // 
            // _waveFormatConfiguration
            // 
            _waveFormatConfiguration.Location = new Point(338, 12);
            _waveFormatConfiguration.MaximumSize = new Size(308, 62);
            _waveFormatConfiguration.MinimumSize = new Size(308, 62);
            _waveFormatConfiguration.Name = "_waveFormatConfiguration";
            _waveFormatConfiguration.Size = new Size(308, 62);
            _waveFormatConfiguration.TabIndex = 5;
            _waveFormatConfiguration.Title = "Wave format";
            // 
            // groupBox1
            // 
            groupBox1.Controls.Add(checkBox1);
            groupBox1.Controls.Add(_txtRemoteAddress);
            groupBox1.Controls.Add(label1);
            groupBox1.Location = new Point(12, 138);
            groupBox1.Name = "groupBox1";
            groupBox1.Size = new Size(320, 54);
            groupBox1.TabIndex = 7;
            groupBox1.TabStop = false;
            groupBox1.Text = "Configuration";
            // 
            // _txtRemoteAddress
            // 
            _txtRemoteAddress.Location = new Point(103, 16);
            _txtRemoteAddress.Name = "_txtRemoteAddress";
            _txtRemoteAddress.Size = new Size(119, 23);
            _txtRemoteAddress.TabIndex = 3;
            _txtRemoteAddress.Text = "127.0.0.1";
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(6, 19);
            label1.Name = "label1";
            label1.Size = new Size(91, 15);
            label1.TabIndex = 0;
            label1.Text = "Remote address";
            // 
            // _ipAddressInvalidError
            // 
            _ipAddressInvalidError.ContainerControl = this;
            // 
            // checkBox1
            // 
            checkBox1.AutoSize = true;
            checkBox1.Location = new Point(228, 18);
            checkBox1.Name = "checkBox1";
            checkBox1.Size = new Size(78, 19);
            checkBox1.TabIndex = 4;
            checkBox1.Text = "Use Effect";
            checkBox1.UseVisualStyleBackColor = true;
            checkBox1.CheckedChanged += checkBox1_CheckedChanged;
            // 
            // MainForm
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(889, 197);
            Controls.Add(groupBox1);
            Controls.Add(_waveFormatConfiguration);
            Controls.Add(_startButton);
            Controls.Add(_waveFormatInfo);
            Controls.Add(_deviceConfiguration);
            Location = new Point(905, 236);
            MaximumSize = new Size(905, 236);
            Name = "MainForm";
            Text = "Form1";
            groupBox1.ResumeLayout(false);
            groupBox1.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)_ipAddressInvalidError).EndInit();
            ResumeLayout(false);
        }

        #endregion

        private Controls.DeviceConfiguration _deviceConfiguration;
        private Controls.WaveFormatInfo _waveFormatInfo;
        private Button _startButton;
        private Controls.WaveFormatConfiguration _waveFormatConfiguration;
        private GroupBox groupBox1;
        private TextBox _txtRemoteAddress;
        private Label label1;
        private ErrorProvider _ipAddressInvalidError;
        private CheckBox checkBox1;
    }
}
