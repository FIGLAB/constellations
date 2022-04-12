#!/usr/bin/python
from protocol import SensorType, StatsFeature

# FFTs
fft_order = [
    "ACCEL_fft", 
    "MICROPHONE_fft",
    "EMI_fft"
]

# Stats
stats_order = [
    "ACCEL_stats",          # Index: 639, 646, 653
    "MICROPHONE_stats",     # Index: 660      
    "EMI_stats",            # Index: 667
    "TEMPERATURE_stats",    #       674
    "HUMIDITY_stats",       #       681
    "BAROMETER_stats",      #       688
    "IRMOTION_stats",       #       695
    "ILLUMINATION_stats",   #       702
    "COLOR_stats",          #       709, 716, 723
    "MAGNETOMETER_stats",   #       730, 737, 744
    "WIFI_RSSI_stats",      #       751
    "GEYE_stats"            #       758
]

def remap_location(location):
    MAPPING = {'01-home-': 'butler-', '02-institution': 'craig-', '03-business': 'qeexo-'}
    for l in MAPPING:
        if (location.startswith(l)):
            location = location.replace('a-office', 'amy-office')
            location = location.replace('c-office', 'chris-office')
            location = location.replace('j-office', 'jeff-office')
            return location.replace(l,MAPPING[l])
            
def reverse_map_location(location):
    MAPPING = {'butler-': '01-home-', 'craig-': '02-institution', 'qeexo-': '03-business'}
    for l in MAPPING:
        if (location.startswith(l)):
            location = location.replace('amy-office', 'a-office')
            location = location.replace('chris-office', 'c-office')
            location = location.replace('jeff-office', 'j-office')
            val = location.replace(l,MAPPING[l])
            return val

def resolve(sensorType,channel,feat):
    indices = {
        SensorType.ACCEL:       640,
        SensorType.MICROPHONE:  661,
        SensorType.EMI:         668,
        SensorType.TEMPERATURE: 675,
        SensorType.HUMIDITY:    682,
        SensorType.BAROMETER:   689,
        SensorType.IRMOTION:    696,
        SensorType.ILLUMINATION: 703,
        SensorType.COLOR:        710,
        SensorType.MAGNETOMETER: 731,
        SensorType.WIFI_RSSI:   752,
        SensorType.GEYE:        759
    }
    if (sensorType in indices):
        return indices[sensorType]+(channel*7)+feat
    return None

fft_count = 128*3 + 128 + 128
stats_count = (7*3) + 7 + 7 + (5*7) + (3*7) + (3*7) + 7 +(64*7)

# Final Feature set: IR, ILLUM, COLOR0, COLOR1, COLOR2, WIFI, TEMP, HUM, BARO

###########################
# High-level Functions
###########################
class EventInstance():
    def __init__(self,session,location,room,sensor,day,start,end):
        self.session = session
        self.location = location
        self.room = room
        self.sensor = sensor # this is the sensor-name, not the MAC address
        self.day = day
        self.start = int(start)
        self.end = int(end)

class DataInstances():
    def __init__(self,session,location,room,day,sensor,data=None,labels=None):
        self.session = session
        self.location = location
        self.room = room
        self.sensor = sensor # this is the sensor-name, not the MAC address
        self.day = day
        self.data = data if data is not None else dict()
        self.labels = labels if labels is not None else []
