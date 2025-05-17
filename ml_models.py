import numpy as np
import tensorflow as tf
import os
import pickle

class BehaviorCloningModel:
    def __init__(self):
        self.model = None
        self.model_path = "models/behavior_cloning"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def build_model(self):
        # Input: vehicle state, sensor readings, path info
        # Output: steering and acceleration commands
        
        # Create a model that uses sensor inputs to predict driving actions
        inputs = tf.keras.Input(shape=(10,)) # Example: 10 features
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        outputs = tf.keras.layers.Dense(2)(x) # [steering, acceleration]
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='adam',
            loss='mse'
        )
        
        return self.model
    
    def train(self, dataset_dir):
        # Load the dataset
        features, labels = self._load_dataset(dataset_dir)
        
        # Create model if it doesn't exist
        if self.model is None:
            self.build_model()
        
        # Train the model
        self.model.fit(
            features, labels,
            epochs=50,
            batch_size=32,
            validation_split=0.2
        )
        
        # Save the model
        self.model.save(self.model_path)
        
    def predict(self, state_vector):
        """Make prediction for a single state vector"""
        if self.model is None:
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except:
                print("No trained model found. Please train first.")
                return [0, 0]  # Default: no steering, no acceleration
                
        # Ensure input is correctly shaped
        input_data = np.array([state_vector])
        prediction = self.model.predict(input_data)[0]
        
        # Return steering and acceleration commands
        return prediction
        
    def _load_dataset(self, dataset_dir):
        """Load and preprocess dataset from CSV files"""
        import pandas as pd
        
        # Load driving data
        df = pd.read_csv(os.path.join(dataset_dir, 'driving_data.csv'))
        
        # Extract features (state)
        features = df[[
            'x', 'y', 'angle', 'speed', 'lateral_offset',
            'nearest_hazard_distance', 'target_path_x', 'target_path_y'
        ]].values
        
        # Extract labels (actions)
        labels = df[['steering_input', 'acceleration_input']].values
        
        return features, labels


class HazardDetectionModel:
    def __init__(self):
        self.model = None
        self.model_path = "models/hazard_detection"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def build_model(self):
        # CNN model for hazard detection from camera images
        # This is simplified - would need proper CNN architecture in practice
        
        # Input: camera images
        inputs = tf.keras.Input(shape=(50, 100, 1)) # Example: 50x100 grayscale
        
        x = tf.keras.layers.Conv2D(16, 3, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        
        # Output: hazard type and confidence
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x) # [none, pothole, speedbreaker]
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, dataset_dir):
        # Load the dataset
        features, labels = self._load_dataset(dataset_dir)
        
        # Create model if it doesn't exist
        if self.model is None:
            self.build_model()
        
        # Train the model
        self.model.fit(
            features, labels,
            epochs=30,
            batch_size=32,
            validation_split=0.2
        )
        
        # Save the model
        self.model.save(self.model_path)
        
    def predict(self, image_data):
        """Make prediction for a single image"""
        if self.model is None:
            try:
                self.model = tf.keras.models.load_model(self.model_path)
            except:
                print("No trained model found. Please train first.")
                return [1, 0, 0]  # Default: no hazard
                
        # Ensure input is correctly shaped
        input_data = np.array([image_data])
        prediction = self.model.predict(input_data)[0]
        
        # Return hazard type probabilities
        return prediction
        
    def _load_dataset(self, dataset_dir):
        """Load and preprocess camera images dataset"""
        import glob
        
        # Get all camera frame files
        image_files = glob.glob(os.path.join(dataset_dir, 'camera_frame_*.npy'))
        
        # Load images
        images = []
        for img_file in image_files:
            img = np.load(img_file)
            # Reshape if needed - depends on how the images are saved
            if len(img.shape) == 2:  # If grayscale, add channel dimension
                img = img.reshape(img.shape[0], img.shape[1], 1)
            images.append(img)
        
        # For demo, we'll generate fake labels
        # In practice, these would be based on hazard_encounters.csv
        labels = np.random.randint(0, 3, size=len(images))
        
        return np.array(images), labels
