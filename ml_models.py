import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle

class BehaviorCloningNetwork(nn.Module):
    def __init__(self, input_size=10):
        super(BehaviorCloningNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # [steering, acceleration]
        )
    
    def forward(self, x):
        return self.network(x)

class BehaviorCloningModel:
    def __init__(self):
        self.model = None
        self.model_path = "models/behavior_cloning.pt"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def build_model(self, input_size=10):
        # Create PyTorch neural network
        self.model = BehaviorCloningNetwork(input_size)
        return self.model
    
    def train(self, dataset_dir):
        # Load the dataset
        features, labels = self._load_dataset(dataset_dir)
        
        # Convert to PyTorch tensors
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.FloatTensor(labels)
        
        # Create model if it doesn't exist
        if self.model is None:
            self.build_model(features.shape[1])
        
        # Setup training
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        epochs = 50
        print(f"Training Behavior Cloning model for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_features, batch_labels in dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
        # Save the model
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def predict(self, state_vector):
        """Make prediction for a single state vector"""
        if self.model is None:
            try:
                input_size = len(state_vector)
                self.model = BehaviorCloningNetwork(input_size)
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
            except:
                print("No trained model found. Please train first.")
                return [0, 0]  # Default: no steering, no acceleration
                
        # Ensure input is correctly shaped and convert to tensor
        input_tensor = torch.FloatTensor(state_vector).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            prediction = self.model(input_tensor).squeeze(0).numpy()
        
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


class HazardDetectionNetwork(nn.Module):
    def __init__(self, input_channels=1):
        super(HazardDetectionNetwork, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Calculate the size after conv layers (depends on input image size)
        # For 50x100 input, after two MaxPool2d layers it becomes 12x25
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 25, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [none, pothole, speedbreaker]
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class HazardDetectionModel:
    def __init__(self):
        self.model = None
        self.model_path = "models/hazard_detection.pt"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
    
    def build_model(self, input_channels=1):
        # CNN model for hazard detection from camera images
        self.model = HazardDetectionNetwork(input_channels)
        return self.model
    
    def train(self, dataset_dir):
        # Load the dataset
        features, labels = self._load_dataset(dataset_dir)
        
        # Convert to PyTorch tensors
        # Reshape features to match CNN input format [batch, channels, height, width]
        features_tensor = torch.FloatTensor(features).permute(0, 3, 1, 2)
        labels_tensor = torch.LongTensor(labels)
        
        # Create model if it doesn't exist
        if self.model is None:
            self.build_model(features_tensor.shape[1])
        
        # Setup training
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epochs = 30
        print(f"Training Hazard Detection model for {epochs} epochs...")
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {100 * correct / total:.2f}%")
        
        # Save the model
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Hazard detection model saved to {self.model_path}")
        
    def predict(self, image_data):
        """Make prediction for a single image"""
        if self.model is None:
            try:
                self.model = HazardDetectionNetwork(input_channels=1)
                self.model.load_state_dict(torch.load(self.model_path))
                self.model.eval()
            except:
                print("No trained hazard detection model found. Please train first.")
                return [1, 0, 0]  # Default: no hazard
                
        # Ensure input is correctly shaped and convert to tensor
        # Convert image to PyTorch expected format [batch, channels, height, width]
        if len(image_data.shape) == 2:  # Add channel dimension if grayscale
            image_data = image_data.reshape(1, image_data.shape[0], image_data.shape[1])
        elif len(image_data.shape) == 3 and image_data.shape[2] == 1:  # Permute if channel is last
            image_data = image_data.transpose((2, 0, 1))
            
        input_tensor = torch.FloatTensor(image_data).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
        
        # Return hazard type probabilities [none, pothole, speedbreaker]
        return probabilities
        
    def _load_dataset(self, dataset_dir):
        """Load and preprocess camera images dataset"""
        import glob
        
        # Get all camera frame files
        image_files = glob.glob(os.path.join(dataset_dir, 'camera_frame_*.npy'))
        
        # Load images
        images = []
        for img_file in image_files:
            img = np.load(img_file)
            # Add channel dimension if the image is grayscale
            if len(img.shape) == 2:
                img = img.reshape(img.shape[0], img.shape[1], 1)
            images.append(img)
        
        # For demo, we'll generate fake labels
        # In practice, these would be based on hazard_encounters.csv
        labels = np.random.randint(0, 3, size=len(images))
        
        return np.array(images), labels
