import csv
import numpy as np
import os
import time

class DataCollector:
    def __init__(self, session_id=None):
        self.data_dir = "datasets"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create unique session ID based on timestamp if not provided
        self.session_id = session_id or f"session_{int(time.time())}"
        self.session_dir = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize different data collectors
        self.driving_data = []
        self.sensor_data = []
        self.hazard_encounters = []
        self.images = []
        
    def record_driving_frame(self, vehicle_state, control_inputs, hazards_nearby, path_info):
        """Record a single frame of driving data"""
        self.driving_data.append({
            'timestamp': time.time(),
            'x': vehicle_state['x'],
            'y': vehicle_state['y'],
            'angle': vehicle_state['angle'],
            'speed': vehicle_state['speed'],
            'lateral_offset': vehicle_state['lateral_offset'],
            'steering_input': control_inputs['steering'],
            'acceleration_input': control_inputs['acceleration'],
            'nearest_hazard_type': hazards_nearby['type'] if hazards_nearby else 'none',
            'nearest_hazard_distance': hazards_nearby['distance'] if hazards_nearby else -1,
            'target_path_x': path_info['target_x'],
            'target_path_y': path_info['target_y'],
            'current_path_segment': path_info['segment_index']
        })
        
    def record_sensor_frame(self, lidar_data, camera_data):
        """Record sensor readings"""
        self.sensor_data.append({
            'timestamp': time.time(),
            'lidar_points': lidar_data,
            'camera_view': camera_data  # This would be a reference to saved image
        })
        
    def save_camera_image(self, image_array, frame_num):
        """Save camera view as image file"""
        # Save numpy array as image file
        filename = f"camera_frame_{frame_num}.npy"
        filepath = os.path.join(self.session_dir, filename)
        np.save(filepath, image_array)
        return filename
        
    def record_hazard_encounter(self, hazard_type, position, vehicle_state, action_taken):
        """Record specific hazard encounters for focused learning"""
        self.hazard_encounters.append({
            'timestamp': time.time(),
            'hazard_type': hazard_type,
            'hazard_x': position[0],
            'hazard_y': position[1],
            'vehicle_x': vehicle_state['x'],
            'vehicle_y': vehicle_state['y'],
            'vehicle_speed': vehicle_state['speed'],
            'vehicle_angle': vehicle_state['angle'],
            'action_steering': action_taken['steering'],
            'action_acceleration': action_taken['acceleration']
        })
        
    def save_datasets(self):
        """Save all collected data to CSV files"""
        # Save driving data
        if self.driving_data:
            with open(os.path.join(self.session_dir, 'driving_data.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.driving_data[0].keys())
                writer.writeheader()
                writer.writerows(self.driving_data)
                
        # Save sensor data references
        if self.sensor_data:
            with open(os.path.join(self.session_dir, 'sensor_data.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.sensor_data[0].keys())
                writer.writeheader()
                writer.writerows(self.sensor_data)
                
        # Save hazard encounters
        if self.hazard_encounters:
            with open(os.path.join(self.session_dir, 'hazard_encounters.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.hazard_encounters[0].keys())
                writer.writeheader()
                writer.writerows(self.hazard_encounters)
                
        print(f"Datasets saved to {self.session_dir}")
        return self.session_dir
