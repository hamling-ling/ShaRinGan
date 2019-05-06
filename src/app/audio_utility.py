import pyaudio

def get_pyaudio_device(p, device_name):
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        dev = p.get_device_info_by_index(i)
        name = dev.get('name')
        if(device_name in name):
            return dev
    return None
