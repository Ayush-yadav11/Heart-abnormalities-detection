import serial
import numpy as np
from time import sleep

class MAX30102Reader:
    def __init__(self, port='COM3', baudrate=115200):
        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=1)
        self.buffer_size = 100  # 1 second of data at 100Hz
        self.sampling_rate = 100  # Hz
        
    def read_sensor(self):
        """Read a single sample from the sensor"""
        if self.serial.in_waiting:
            try:
                line = self.serial.readline().decode('utf-8').strip()
                # Expecting data in format: "IR,RED"
                ir_val, red_val = map(int, line.split(','))
                return ir_val
            except (ValueError, IndexError):
                return None
        return None
    
    def get_signal_window(self, duration=1):
        """Collect signal data for specified duration in seconds"""
        samples_needed = int(self.sampling_rate * duration)
        signal_data = []
        
        while len(signal_data) < samples_needed:
            sample = self.read_sensor()
            if sample is not None:
                signal_data.append(sample)
            sleep(1/self.sampling_rate)  # Control sampling rate
            
        return np.array(signal_data)
    
    def close(self):
        """Close the serial connection"""
        if self.serial.is_open:
            self.serial.close()