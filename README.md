# Autonomous Vehicle and Robotics Simulation

A Python-based simulation for autonomous vehicle control with PyTorch machine learning capabilities.

## Features

- Real-time vehicle simulation with physics
- Automatic and manual control modes
- Hazard detection and avoidance
- Data collection for machine learning
- Machine learning model integration
  - Behavior cloning for autonomous driving
  - Hazard detection using computer vision

## Controls

- **Arrow Keys**: Control vehicle in manual mode
- **M**: Switch between automatic and manual modes
- **R**: Toggle data recording
- **L**: Toggle ML-based control (when available)
- **T**: Train ML models from collected data
- **ESC**: Quit

## Machine Learning

The project supports two ML approaches:
1. **Behavior Cloning**: Learn driving behavior from human demonstrations
2. **Hazard Detection**: Identify hazards (potholes, speed breakers) from simulated camera data

## Getting Started

1. Make sure you have Python 3.7+ installed
2. Create and activate the virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the simulation:
   ```
   ./run.sh  # On Windows: run.bat
   ```
   
   Alternatively:
   ```
   source venv/bin/activate
   python autonomous_vehicle_and_robotics.py
   ```

## Data Collection

Data is collected in the `datasets` directory when recording is active. Each session includes:
- `driving_data.csv`: Vehicle state and control inputs
- `sensor_data.csv`: Sensor readings
- `hazard_encounters.csv`: Specific hazard interaction data
- Camera frame images

## Training ML Models

### Using the Simulation Interface
1. Start recording data by pressing **R**
2. Drive the vehicle to collect training data
3. Stop recording by pressing **R** again
4. Train models by pressing **T**
5. Enable ML-based control by pressing **L**

### Using the Training Utility
The project includes a dedicated training utility:

```
# List available datasets
./train_models.py --list

# Train using the latest dataset
./train_models.py --latest

# Train a specific model type
./train_models.py --latest --models behavior
./train_models.py --latest --models hazard

# Train using a specific dataset
./train_models.py --dataset session_1747503003
```