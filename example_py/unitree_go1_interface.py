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
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (w, x, y, z). Shape is (N, 4).
        v: The vector in (x, y, z). Shape is (N, 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (N, 3).
    """
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

class UnitreeGo1Interface:
    def __init__(
            self,
            dt=0.002,
            max_base_vel=0.2,
            device='cpu'
            
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
        self.last_action = None

        # Create command and state objects
        self.cmd = sdk.LowCmd()
        self.state = sdk.HighState() # High state allows access to both joint level values and body-level info
        
        # Initialize connection to the robot
        self._IP = "192.168.123.10"
        self.udp = sdk.UDP(LOWLEVEL, 8080, self._IP, 8007)
        self._safe = sdk.Safety(sdk.LeggedType.Go1)
        self.udp.InitCmdData(self.cmd)
    
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

        return projected_gravity

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
        self.udp.Recv()
        self.udp.GetRecv(self.state)

        # Base linear velocity
        base_lin_vel = torch.tensor(state.velocity, device=self.device).reshape(1, 3)
        # Base angular velocity
        base_ang_vel = torch.tensor(state.imu.gyroscope, device=self.device).reshape(1, 3)
        # Base projected gravity vector
        projected_gravity = self._compute_projected_gravity() # Already returns a tensor
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
        ])

        return self.last_obs

    def set_joint_targets(self, joint_targets):
        """
        Set the joint targets for the robot.
        
        :param joint_targets: List of target positions for each joint.
        """
        if len(joint_targets) != len(self.current_joint_targets):
            raise ValueError(f"Expected {len(self.current_joint_targets)} joint targets, got {len(joint_targets)}")
        
        # Send joint targets to the SDK
        self.sdk_interface.send_joint_targets(joint_targets)
        
        # Store the new joint targets internally
        self.current_joint_targets = joint_targets

    def set_base_velocity(self, base_velocity):
        """
        Set the desired base velocity for the robot (e.g., forward, sideways, rotational velocity).
        
        :param base_velocity: List or tuple with base velocity in [vx, vy, v_yaw].
        """
        if len(base_velocity) != 3:
            raise ValueError("Base velocity should be a list or tuple with 3 values: [vx, vy, v_yaw]")
        
        # Clamp the velocity to max limits
        clamped_velocity = [
            min(max(base_velocity[0], -self.max_base_speed[0]), self.max_base_speed[0]),  # vx
            min(max(base_velocity[1], -self.max_base_speed[1]), self.max_base_speed[1]),  # vy
            min(max(base_velocity[2], -self.max_base_speed[2]), self.max_base_speed[2])   # v_yaw
        ]
        
        # Send the velocity to the SDK
        self.sdk_interface.send_base_velocity(clamped_velocity)
        
        # Store the current base velocity
        self.current_base_velocity = clamped_velocity

    def set_default_joint_values(self, default_joint_values):
        """
        Set the default joint values for the robot.
        
        :param default_joint_values: List or dictionary of default joint values.
        """
        if len(default_joint_values) != len(self.current_joint_targets):
            raise ValueError(f"Expected {len(self.current_joint_targets)} joint values, got {len(default_joint_values)}")
        
        self.default_joint_values = default_joint_values

    def reset_to_default_joint_values(self):
        """
        Reset the robot's joint targets to the default joint values.
        """
        self.set_joint_targets(self.default_joint_values)

    def set_max_base_speed(self, max_base_speed):
        """
        Set the maximum base speed for the robot.
        
        :param max_base_speed: List with max speeds for [vx, vy, v_yaw].
        """
        if len(max_base_speed) != 3:
            raise ValueError("Max base speed should be a list with 3 values: [vx, vy, v_yaw]")
        
        self.max_base_speed = max_base_speed

    def get_current_state(self):
        """
        Return the current state of the robot including joint targets and base velocity.
        :return: Dictionary containing current joint targets and base velocity.
        """
        return {
            'joint_targets': self.current_joint_targets,
            'base_velocity': self.current_base_velocity
        }

# Example usage
# sdk = UnitreeSDK()  # Hypothetical SDK interface instance
# robot_interface = UnitreeGo1Interface(sdk)
# robot_interface.set_joint_targets([0.1] * 12)
# robot_interface.set_base_velocity([0.2, 0.0, 0.1])



def _get_observations(state, device):
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

    # Base linear velocity
    base_lin_vel = torch.tensor(state.velocity, device=device).reshape(1, 3)
    # Base angular velocity
    base_ang_vel = torch.tensor(state.imu.gyroscope, device=device).reshape(1, 3)
    # Base projected gravity vector
    # Target velocity commands
    # Relative joint positions wrt default
    # Relative joint velocities wrt default
    # Previous action


    # Get the observations
    obs = 0


    return obs


if __name__ == '__main__':

    # Indexes of each leg's motors
    jointIdx = {'FR_0':0, 'FR_1':1, 'FR_2':2,
         'FL_0':3, 'FL_1':4, 'FL_2':5, 
         'RR_0':6, 'RR_1':7, 'RR_2':8, 
         'RL_0':9, 'RL_1':10, 'RL_2':11 }
    
    # Max values for position and velocity
    PosStopF  = math.pow(10,9)
    VelStopF  = 16000.0
    # Addres to choose between high level or low level control
    HIGHLEVEL = 0xee
    LOWLEVEL  = 0xff

    # Set hyperparameters
    # TODO: Make it so that it controls every joint
    sin_mid_q = [0.0, 1.2, -2.0]
    dt = 0.002
    qInit = [0, 0, 0]
    qDes = [0, 0, 0]
    sin_count = 0
    rate_count = 0
    Kp = [0, 0, 0]
    Kd = [0, 0, 0]

    # Create UDP connection to the robot
    udp = sdk.UDP(LOWLEVEL, 8080, "192.168.123.10", 8007)
    safe = sdk.Safety(sdk.LeggedType.Go1)
    
    # Create command and state objects
    cmd = sdk.LowCmd()
    state = sdk.HighState() # High state allows access to both joint level values and body-level info
    udp.InitCmdData(cmd)

    Tpi = 0
    motiontime = 0
    while True:
        # Set dt
        time.sleep(dt)
        motiontime += 1

        # print(motiontime)
        # print(state.imu.rpy[0])
        
        # Receive robot's current state
        udp.Recv()
        udp.GetRecv(state)
        
        # Only access after first iteration so that the robot is initialized
        if( motiontime >= 0):

            # first, get record initial position
            if( motiontime >= 0 and motiontime < 10):
                qInit[0] = state.motorState[jointIdx['FR_0']].q
                qInit[1] = state.motorState[jointIdx['FR_1']].q
                qInit[2] = state.motorState[jointIdx['FR_2']].q
            
            # second, move to the origin point of a sine movement with Kp Kd
            if( motiontime >= 10 and motiontime < 400):
                rate_count += 1
                rate = rate_count/200.0                       # needs count to 200
                Kp = [5, 5, 5]
                Kd = [1, 1, 1]
                # Kp = [20, 20, 20]
                # Kd = [2, 2, 2]
                
                qDes[0] = jointLinearInterpolation(qInit[0], sin_mid_q[0], rate)
                qDes[1] = jointLinearInterpolation(qInit[1], sin_mid_q[1], rate)
                qDes[2] = jointLinearInterpolation(qInit[2], sin_mid_q[2], rate)
            
            # last, do sine wave
            freq_Hz = 1
            # freq_Hz = 5
            freq_rad = freq_Hz * 2* math.pi
            t = dt*sin_count
            if( motiontime >= 400):
                sin_count += 1
                # sin_joint1 = 0.6 * sin(3*M_PI*sin_count/1000.0)
                # sin_joint2 = -0.9 * sin(3*M_PI*sin_count/1000.0)
                sin_joint1 = 0.6 * math.sin(t*freq_rad)
                sin_joint2 = -0.9 * math.sin(t*freq_rad)
                qDes[0] = sin_mid_q[0]
                qDes[1] = sin_mid_q[1] + sin_joint1
                qDes[2] = sin_mid_q[2] + sin_joint2
            
            # Set the motor commands for each joint
            cmd.motorCmd[jointIdx['FR_0']].q = qDes[0]
            cmd.motorCmd[jointIdx['FR_0']].dq = 0
            cmd.motorCmd[jointIdx['FR_0']].Kp = Kp[0]
            cmd.motorCmd[jointIdx['FR_0']].Kd = Kd[0]
            cmd.motorCmd[jointIdx['FR_0']].tau = -0.65

            cmd.motorCmd[jointIdx['FR_1']].q = qDes[1]
            cmd.motorCmd[jointIdx['FR_1']].dq = 0
            cmd.motorCmd[jointIdx['FR_1']].Kp = Kp[1]
            cmd.motorCmd[jointIdx['FR_1']].Kd = Kd[1]
            cmd.motorCmd[jointIdx['FR_1']].tau = 0.0

            cmd.motorCmd[jointIdx['FR_2']].q =  qDes[2]
            cmd.motorCmd[jointIdx['FR_2']].dq = 0
            cmd.motorCmd[jointIdx['FR_2']].Kp = Kp[2]
            cmd.motorCmd[jointIdx['FR_2']].Kd = Kd[2]
            cmd.motorCmd[jointIdx['FR_2']].tau = 0.0
            # cmd.motorCmd[jointIdx['FR_2']].tau = 2 * sin(t*freq_rad)


        if(motiontime > 10):
            safe.PowerProtect(cmd, state, 1)

        # Send the motor commands to the robot
        udp.SetSend(cmd)
        udp.Send()
