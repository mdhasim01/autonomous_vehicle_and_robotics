# Autonomous Vehicle and Robotics Simulation

A Python-based simulation for autonomous vehicle control with machine learning capabilities.

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
2. Install required packages:
   ```
   pip install pygame numpy tensorflow
   ```
3. Run the simulation:
   ```
   python autonomous_vehicle_and_robotics.py
   ```

## Data Collection

Data is collected in the `datasets` directory when recording is active. Each session includes:
- `driving_data.csv`: Vehicle state and control inputs
- `sensor_data.csv`: Sensor readings
- `hazard_encounters.csv`: Specific hazard interaction data
- Camera frame images