#!/usr/bin/python

import sys
import time
import math
import numpy as np
import torch

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion along the last dimension of q and v.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c

class UnitreeGo1Interface:
    def __init__(
            self,
            dt=0.002,
            max_base_vel=0.2,
            device='cpu',
    ):
        """
        Initialize the Unitree Go1 interface.
        
        """

        ### SDK propietary values. see: https://github.com/unitreerobotics/unitree_legged_sdk/blob/go1/example_py/example_position.py
        # Indexes of each leg's motors
        self._jointIdx = {'FR_0':0, 'FR_1':1, 'FR_2':2,
             'FL_0':3, 'FL_1':4, 'FL_2':5, 
             'RR_0':6, 'RR_1':7, 'RR_2':8, 
             'RL_0':9, 'RL_1':10, 'RL_2':11 }
        self._PosStopF  = math.pow(10,9)
        self._VelStopF  = 16000.0
        self._HIGHLEVEL = 0xee
        self._LOWLEVEL  = 0xff

        # Robot-related initial conditions
        self.default_joint_pos = {
             'FR_0':-0.1, 'FR_1':0.8, 'FR_2':-1.5,
             'FL_0':0.1, 'FL_1':0.8, 'FL_2':-1.5, 
             'RR_0':-0.1, 'RR_1':1.0, 'RR_2':-1.5, 
             'RL_0':0.1, 'RL_1':1.0, 'RL_2': -1.5}

        # Adjustable parameters
        self.dt = dt
        self.max_base_vel = max_base_vel
        self.Kp = np.ones(12) * 5.0
        self.Kd = np.ones(12) * 1.0

        # Target base velocity (Vx, Vy, Wz)
        self.velocity_target = [0.0, 0.0, 0.0]

        # Set device for torch tensors
        self.device = device

        # Buffers to store previous observations and actions
        self.last_obs = None
        self.last_action = torch.zeros(12, device=self.device)

        # Create command and state objects
        self.cmd = sdk.LowCmd()
        self.state = sdk.HighState() # High state allows access to both joint-level values and body-level info
        
        # Initialize connection to the robot
        print("Initializing UDP connection to the robot...")
        self._IP = "192.168.123.10"
        self.udp = sdk.UDP(self._LOWLEVEL, 8080, self._IP, 8007)
        self._safe = sdk.Safety(sdk.LeggedType.Go1)
        self.udp.InitCmdData(self.cmd)
        print("UDP connection to the robot initialized.")

    ###### PRIVATE METHODS ######
    
    def _get_relative_joint_positions(self):
        # Compute relative joint position vector from robot state provided by the SDK
        joint_pos_rel = []
        for idx in self._jointIdx.values():
            joint_pos_rel.append(self.state.motorState[idx].q - self.default_joint_pos[idx])
        
        return joint_pos_rel
    
    def _get_relative_joint_velocities(self):
        # Compute relative joint velocity vector from robot state provided by the SDK
        # We assume that default joint velocities are zero
        joint_vel_rel = []
        for idx in self._jointIdx.values():
            joint_vel_rel.append(self.state.motorState[idx].dq)
        
        return joint_vel_rel
    
    def _compute_projected_gravity(self):
        # Compute the projected gravity vector from robot state provided by the SDK
        gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device).reshape(1, 3)
        orientation = torch.tensor(self.state.imu.quaternion, device=self.device).reshape(1, 4)
        projected_gravity = quat_rotate_inverse(orientation, gravity)

        return projected_gravity.reshape(1, 3)
    
    def _update_robot_state(self):
        # Update robot state via UDP
        self.udp.Recv()
        self.udp.GetRecv(self.state)

    def _parse_action(self, action):
        # Parse the action tensor to reorder the joint commands in order to match the SDK's expected format
        # The SDK expects the joint commands in the following order: 
        # [FR_0, FR_1, FR_2, FL_0, FL_1, FL_2, RR_0, RR_1, RR_2, RL_0, RL_1, RL_2]
        # But IsaacLab's policy outputs the joint commands in the following order:
        # [FL_0, FR_0, RL_0, RR_0, FL_1, FR_1, RL_1, RR_1, FL_2, FR_2, RL_2, RR_2]
        self.action_dict = {}

        # Convert action tensor to array
        action = action.detach().numpy().squeeze()

        # Reorder the joint commands
        self.action_dict['FR_0'] = action[1]
        self.action_dict['FR_1'] = action[5]
        self.action_dict['FR_2'] = action[9]
        self.action_dict['FL_0'] = action[0]
        self.action_dict['FL_1'] = action[4]
        self.action_dict['FL_2'] = action[8]
        self.action_dict['RR_0'] = action[3]
        self.action_dict['RR_1'] = action[7]
        self.action_dict['RR_2'] = action[11]
        self.action_dict['RL_0'] = action[2]
        self.action_dict['RL_1'] = action[6]
        self.action_dict['RL_2'] = action[10]

            

    ###### PUBLIC METHODS ######

    def compute_observation(self):
        """
        Get the observations from the robot's state. the observation vector needs to follow the same order that the one for
        the trained policy in IsaacLab. 

        Parameters:
        - state: The robot's current state.
        - device: Device in which the policy is deployed (cpu/gpu). To return the observation vector in same device as policy

        Returns:
        - obs: The observations.

        Example:
        obs = _get_observations(low_state, state, device)
        """

        # Update robot state
        self._update_robot_state()

        # Base linear velocity
        base_lin_vel = torch.tensor(self.state.velocity, device=self.device).reshape(1, 3)
        # Base angular velocity
        base_ang_vel = torch.tensor(self.state.imu.gyroscope, device=self.device).reshape(1, 3)
        # Base projected gravity vector
        projected_gravity = self._compute_projected_gravity() # Already returns staa tensor
        # Target velocity commands
        velocity_command = torch.tensor(self.velocity_target, device=self.device).reshape(1, 3)
        # Relative joint positions wrt default
        joint_pos_rel = torch.tensor(self._get_relative_joint_positions(), device=self.device).reshape(1, 12)        
        # Relative joint velocities wrt default
        joint_vel_rel = torch.tensor(self._get_relative_joint_velocities(), device=self.device).reshape(1, 12)
        # Previous action
        last_action = torch.tensor(self.last_action, device=self.device).reshape(1, 12)

        # Concatenate all observations
        self.last_obs = torch.cat([
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            velocity_command,
            joint_pos_rel,
            joint_vel_rel,
            last_action
        ], device=self.device)

        return self.last_obs

    def set_action(self, action):
        """
        Set the action to be executed by the robot.
        
        :param action: The action tensor.
        """
        # Parse the action tensor to match the SDK's expected format
        self.action_dict = self._parse_action(action)

        # Set the motor commands for each joint
        for joint_name, idx in self._jointIdx.items():
            self.cmd.motorCmd[idx].q = self.action_dict[joint_name]
            self.cmd.motorCmd[idx].dq = 0
            self.cmd.motorCmd[idx].Kp = self.Kp[idx]
            self.cmd.motorCmd[idx].Kd = self.Kd[idx]
            self.cmd.motorCmd[idx].tau = 0.0

        # Store the last action
        self.last_action = action

    def send_action(self):
        """
        Send the motor commands to the robot.
        """
        # Send the motor commands to the robot
        self.udp.SetSend(self.cmd)
        self.udp.Send()  

    def set_velocity_target(self, velocity_target):
        """
        Set the target velocity for the robot's base.
        
        :param velocity_target: The target velocity [Vx, Vy, Wz].
        """
        # Clip the target velocity to the maximum allowed
        self.velocity_target = np.clip(velocity_target, -self.max_base_vel, self.max_base_vel)
    
    def set_power_limit(self, power_limit: int = 1):
        """
        Set the power limit for the robot's motors.
        
        :param power_limit: The power limit (1 to 10).
        """
        if power_limit < 1 or power_limit > 10:
            raise ValueError("Power limit must be between 1 and 10")
        
        self._safe.PowerProtect(self.cmd, self.state, power_limit)

if __name__ == '__main__':
    '''
    This is an example of how to use the Unitree Go1 interface.
    '''

    # Create the Unitree Go1 interface. this will initialize the connection to the robot
    go1_interface = UnitreeGo1Interface(
        dt=0.002, # Time step in seconds
        max_base_vel=0.2, # Maximum base velocity in m/s
        device='cpu' # Device in which the policy is deployed (cpu/cuda)
    )

    # Set velocity target (TODO: make this dynamic with keyboard or joystick)
    Vx = 0.0
    Vy = 0.0
    Wz = 0.0
    go1_interface.set_velocity_target([ Vx, Vy, Wz ])

    motiontime = 0
    while True:
        # Set dt
        time.sleep(go1_interface.dt)
        motiontime += 1
        
        # Receive robot's current state
        obs = go1_interface.compute_observation()
        
        # Only access after first iteration so that the robot is initialized
        if( motiontime >= 0):
            # Get the action from the policy
            action = torch.zeros(12, device=go1_interface.device) # Dummy action. Replace this with the action from the policy
            go1_interface.set_action(action)

        if(motiontime > 10):
            go1_interface.set_power_limit(1) # Limits power to the motors to <param> * 10%

        # Send the motor commands to the robot
        go1_interface.send_action()
