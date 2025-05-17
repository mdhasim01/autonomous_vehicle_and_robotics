#!/usr/bin/env python3
import os
import glob
import time
import argparse
from ml_models import BehaviorCloningModel, HazardDetectionModel

def list_datasets():
    """List all available datasets."""
    if not os.path.exists("datasets"):
        print("No datasets directory found.")
        return []
    
    session_dirs = [d for d in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", d))]
    if not session_dirs:
        print("No datasets found.")
        return []
    
    session_dirs = sorted(session_dirs, key=lambda x: os.path.getmtime(os.path.join("datasets", x)), reverse=True)
    print("Available datasets:")
    for i, session in enumerate(session_dirs):
        # Get creation time and format it
        timestamp = os.path.getmtime(os.path.join("datasets", session))
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        
        # Get dataset size statistics
        driving_data = os.path.exists(os.path.join("datasets", session, "driving_data.csv"))
        camera_frames = len(glob.glob(os.path.join("datasets", session, "camera_frame_*.npy")))
        hazard_data = os.path.exists(os.path.join("datasets", session, "hazard_encounters.csv"))
        
        print(f"{i+1}. {session} - {time_str}")
        print(f"   Driving data: {'✓' if driving_data else '✗'}, Camera frames: {camera_frames}, Hazard data: {'✓' if hazard_data else '✗'}")
    
    return session_dirs

def train_models(dataset_path, models=None):
    """Train specified models using the dataset."""
    if not models:
        models = ["behavior", "hazard"]
    
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} does not exist.")
        return
    
    print(f"Training models using dataset: {dataset_path}")
    
    if "behavior" in models:
        print("\nTraining Behavior Cloning Model...")
        try:
            bc_model = BehaviorCloningModel()
            bc_model.train(dataset_path)
            print("✓ Behavior Cloning model trained successfully")
        except Exception as e:
            print(f"✗ Error training Behavior Cloning model: {str(e)}")
    
    if "hazard" in models:
        print("\nTraining Hazard Detection Model...")
        try:
            hazard_model = HazardDetectionModel()
            hazard_model.train(dataset_path)
            print("✓ Hazard Detection model trained successfully")
        except Exception as e:
            print(f"✗ Error training Hazard Detection model: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Train ML models for autonomous vehicle simulation")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--dataset", type=str, help="Dataset to use for training (provide session directory name)")
    parser.add_argument("--latest", action="store_true", help="Use the latest dataset for training")
    parser.add_argument("--models", type=str, choices=["behavior", "hazard", "both"], default="both", 
                        help="Which models to train (behavior, hazard, or both)")
    
    args = parser.parse_args()
    
    if args.list or (not args.dataset and not args.latest):
        session_dirs = list_datasets()
        if not session_dirs:
            return
    
    models_to_train = ["behavior", "hazard"] if args.models == "both" else [args.models]
    
    if args.latest:
        # Find the latest dataset
        session_dirs = [os.path.join("datasets", d) for d in os.listdir("datasets") 
                      if os.path.isdir(os.path.join("datasets", d))]
        if not session_dirs:
            print("No datasets found.")
            return
        
        latest_dataset = max(session_dirs, key=os.path.getmtime)
        train_models(latest_dataset, models_to_train)
    
    elif args.dataset:
        # Use the specified dataset
        dataset_path = os.path.join("datasets", args.dataset) if not os.path.dirname(args.dataset) else args.dataset
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found.")
            return
        
        train_models(dataset_path, models_to_train)

if __name__ == "__main__":
    main()
