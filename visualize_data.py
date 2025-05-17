#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def visualize_dataset(dataset_path):
    """Visualize dataset statistics and key metrics."""
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist.")
        return
    
    # Check for driving data
    driving_data_path = os.path.join(dataset_path, "driving_data.csv")
    if not os.path.exists(driving_data_path):
        print(f"No driving data found in {dataset_path}")
        return
    
    print(f"Analyzing dataset: {os.path.basename(dataset_path)}")
    
    try:
        # Load driving data
        df = pd.read_csv(driving_data_path)
        
        # Print basic statistics
        print("\nDataset Statistics:")
        print(f"Total frames: {len(df)}")
        print(f"Unique hazard types: {df['nearest_hazard_type'].unique()}")
        print(f"Average speed: {df['speed'].mean():.2f}")
        print(f"Max speed: {df['speed'].max():.2f}")
        
        # Setup plotting
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle(f"Dataset Analysis: {os.path.basename(dataset_path)}", fontsize=16)
    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return
    
    # Plot vehicle path
    axes[0][0].scatter(df['x'], df['y'], c=df.index, s=5, cmap='viridis')
    axes[0][0].set_title('Vehicle Path')
    axes[0][0].set_xlabel('X position')
    axes[0][0].set_ylabel('Y position')
    axes[0][0].grid(True)
    
    # Plot speed over time
    axes[0][1].plot(df.index, df['speed'], 'b-')
    axes[0][1].set_title('Speed Profile')
    axes[0][1].set_xlabel('Frames')
    axes[0][1].set_ylabel('Speed')
    axes[0][1].grid(True)
    
    # Plot steering input
    axes[1][0].plot(df.index, df['steering_input'], 'g-')
    axes[1][0].set_title('Steering Input')
    axes[1][0].set_xlabel('Frames')
    axes[1][0].set_ylabel('Steering Value')
    axes[1][0].grid(True)
    
    # Plot acceleration input
    axes[1][1].plot(df.index, df['acceleration_input'], 'r-')
    axes[1][1].set_title('Acceleration Input')
    axes[1][1].set_xlabel('Frames')
    axes[1][1].set_ylabel('Acceleration Value')
    axes[1][1].grid(True)
    
    # Plot nearest hazard distance
    axes[2][0].plot(df.index, df['nearest_hazard_distance'], 'purple')
    axes[2][0].set_title('Nearest Hazard Distance')
    axes[2][0].set_xlabel('Frames')
    axes[2][0].set_ylabel('Distance')
    axes[2][0].grid(True)
    
    # Plot lateral offset
    axes[2][1].plot(df.index, df['lateral_offset'], 'orange')
    axes[2][1].set_title('Lateral Offset')
    axes[2][1].set_xlabel('Frames')
    axes[2][1].set_ylabel('Offset')
    axes[2][1].grid(True)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the visualization
    output_path = os.path.join(dataset_path, "visualization.png")
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    
    # Try to display the plot if in interactive environment
    try:
        plt.show()
    except:
        pass

def visualize_camera_data(dataset_path, num_samples=5):
    """Visualize sample camera frames from the dataset."""
    import glob
    import matplotlib.image as mpimg
    
    try:
        # Find camera frames
        camera_frames = glob.glob(os.path.join(dataset_path, "camera_frame_*.npy"))
        
        if not camera_frames:
            print(f"No camera frames found in {dataset_path}")
            return
        
        print(f"Found {len(camera_frames)} camera frames, displaying {min(num_samples, len(camera_frames))} samples")
        
        # Sample frames evenly across the dataset
        if len(camera_frames) > num_samples:
            indices = np.linspace(0, len(camera_frames)-1, num_samples, dtype=int)
            camera_frames = [camera_frames[i] for i in indices]
        
        # Display frames
        fig, axes = plt.subplots(1, len(camera_frames), figsize=(4*len(camera_frames), 4))
        if len(camera_frames) == 1:
            axes = [axes]
    except Exception as e:
        print(f"Error setting up camera visualization: {str(e)}")
        return
    
    for i, frame_path in enumerate(camera_frames):
        try:
            print(f"Loading camera frame: {frame_path}")
            frame = np.load(frame_path)
            print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            axes[i].imshow(frame, cmap='gray')
            axes[i].set_title(f"Frame {os.path.basename(frame_path)}")
            axes[i].axis('off')
        except Exception as e:
            print(f"Error loading {frame_path}: {str(e)}")
            # Add error message to plot
            axes[i].text(0.5, 0.5, f"Error: {str(e)[:20]}...", 
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[i].transAxes, color='red')
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = os.path.join(dataset_path, "camera_samples.png")
    try:
        plt.savefig(output_path)
        print(f"Camera visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {str(e)}")
    
    # Try to display the plot if in interactive environment
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Visualize dataset statistics")
    parser.add_argument("--dataset", type=str, help="Dataset to analyze (session directory name)")
    parser.add_argument("--latest", action="store_true", help="Analyze the latest dataset")
    parser.add_argument("--camera", action="store_true", help="Visualize sample camera frames")
    
    args = parser.parse_args()
    
    if not args.dataset and not args.latest:
        parser.print_help()
        return
    
    if args.latest:
        # Find the latest dataset
        datasets_dir = "datasets"
        if not os.path.exists(datasets_dir):
            print("No datasets directory found.")
            return
            
        session_dirs = [os.path.join(datasets_dir, d) for d in os.listdir(datasets_dir) 
                       if os.path.isdir(os.path.join(datasets_dir, d))]
        if not session_dirs:
            print("No datasets found.")
            return
            
        dataset_path = max(session_dirs, key=os.path.getmtime)
    else:
        # Use the specified dataset
        dataset_path = os.path.join("datasets", args.dataset) if not os.path.dirname(args.dataset) else args.dataset
    
    # Run visualizations
    visualize_dataset(dataset_path)
    
    if args.camera:
        visualize_camera_data(dataset_path)

if __name__ == "__main__":
    main()
