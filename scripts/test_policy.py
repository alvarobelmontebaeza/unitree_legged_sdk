import torch
import numpy as np
import time
from modules.unitree_go1_interface import UnitreeGo1Interface
import argparse


# Create an argument parser
parser = argparse.ArgumentParser(description='Test Policy')

# Add arguments
parser.add_argument('--device', type=str, default='cpu', help='Device to run the code (cpu/cuda)')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')

# Parse the arguments
args = parser.parse_args()
device = args.device
model_path = args.model_path

# Create an instance of the UnitreeGo1Interface class
go1_interface = UnitreeGo1Interface(
    dt = 0.002,
    max_base_vel=0.2,
    device=device
)

# Load the trained model
policy = torch.load(args.model_path)

# Set the model to evaluation mode
policy.eval()

# Set velocity target
Vx = 0.0
Vy = 0.0
Wz = 0.0
go1_interface.set_velocity_target([ Vx, Vy, Wz ])

# Initialize the state
obs = go1_interface.compute_observation()

motionTime = 0

# Main loop
while True:
    with torch.inference_mode():
        time.sleep(go1_interface.dt)
        motionTime += 1
        
        # Get observations from the unitree_go1_interface class
        obs = go1_interface.compute_observation()
        if motionTime >= 0:
            # Pass the tensor observations through the model to get the predicted actions
            actions = policy(obs)
            # Convert the predicted actions to the required format (if needed)
            actions = go1_interface.set_action(actions)

        if motionTime > 10:
            go1_interface.set_power_limit(1)

        # Sends the currently set actions robot via UDP connection
        go1_interface.send_action()