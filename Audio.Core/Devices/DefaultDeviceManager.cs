﻿using NAudio.CoreAudioApi;

namespace Audio.Devices
{
    internal abstract class DefaultDeviceManager : IDeviceManager
    {
        private readonly DataFlow _dataFlow;

        public DefaultDeviceManager(DataFlow dataFlow)
        {
            _dataFlow = dataFlow;
        }

        public virtual IEnumerable<MMDevice> GetDevices()
        {
            return DevicesFactory.GetDevices(_dataFlow);
        }

        public virtual bool TryGetDevice(string deviceId, out MMDevice? outDevice)
        {
            var devices = GetDevices();

            if (!devices.Any())
            {
                outDevice = default;
                return false;
            }

            if (string.IsNullOrEmpty(deviceId))
            {
                outDevice = devices.First();
                return true;
            }
            outDevice = devices.FirstOrDefault(d => d.ID == deviceId);
            if (outDevice == null)
            {
                return false;
            }
            return true;
        }
    }
}