#!/usr/bin/python

import sys
import time
import math
import numpy as np

sys.path.append('../lib/python/amd64')
import robot_interface as sdk

def jointLinearInterpolation(initPos, targetPos, rate):
    """
    Performs linear interpolation between the initial position (initPos) and the target position (targetPos) based on the given rate.

    Parameters:
    - initPos: The initial position.
    - targetPos: The target position.
    - rate: The interpolation rate, ranging from 0.0 to 1.0.

    Returns:
    - p: The interpolated position.

    Example:
    initPos = 0
    targetPos = 10
    rate = 0.5
    p = jointLinearInterpolation(initPos, targetPos, rate)
    # p = 5
    """

    rate = np.fmin(np.fmax(rate, 0.0), 1.0)
    p = initPos*(1-rate) + targetPos*rate
    return p

def _get_observations(low_state, high_state):
    """
    Get the observations from the robot's state.

    Parameters:
    - low_state: The robot's low state.
    - high_state: The robot's high state.

    Returns:
    - obs: The observations.

    Example:
    obs = _get_observations(low_state, high_state)
    """
    # Base linear velocity
    # Base angular velocity
    # Base projected gravity vector
    # Target velocity commands
    # Relative joint positions wrt default
    # Relative joint velocities wrt default
    # Previous action
    # Get the joint positions
    joint_positions = np.array([low_state.motorState[i].q for i in range(12)])

    # Get the joint velocities
    joint_velocities = np.array([low_state.motorState[i].dq for i in range(12)])

    # Get the joint torques
    joint_torques = np.array([low_state.motorState[i].tau for i in range(12)])

    # Get the IMU orientation
    imu_orientation = np.array([high_state.imu.rpy[i] for i in range(3)])

    # Get the IMU angular velocity
    imu_angular_velocity = np.array([high_state.imu.gyro[i] for i in range(3)])

    # Get the IMU linear acceleration
    imu_linear_acceleration = np.array([high_state.imu.acc[i] for i in range(3)])

    # Get the body velocity
    body_velocity = np.array([high_state.velocity[i] for i in range(3)])

    # Get the body angular velocity
    body_angular_velocity = np.array([high_state.angularVelocity[i] for i in range(3)])

    # Get the body linear acceleration
    body_linear_acceleration = np.array([high_state.linearAcc[i] for i in range(3)])

    # Get the time
    time = high_state.time

    # Get the observations
    obs = {
        'joint_positions': joint_positions,
        'joint_velocities': joint_velocities,
        'joint_torques': joint_torques,
        'imu_orientation': imu_orientation,
        'imu_angular_velocity': imu_angular_velocity,
        'imu_linear_acceleration': imu_linear_acceleration,
        'body_velocity': body_velocity,
        'body_angular_velocity': body_angular_velocity,
        'body_linear_acceleration': body_linear_acceleration,
        'time': time
    }

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
    low_state = sdk.LowState() # For joint related values
    high_state = sdk.HighState() # For body velocity
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
        udp.GetRecv(low_state)
        
        # Only access after first iteration so that the robot is initialized
        if( motiontime >= 0):

            # first, get record initial position
            if( motiontime >= 0 and motiontime < 10):
                qInit[0] = low_state.motorState[jointIdx['FR_0']].q
                qInit[1] = low_state.motorState[jointIdx['FR_1']].q
                qInit[2] = low_state.motorState[jointIdx['FR_2']].q
            
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
            safe.PowerProtect(cmd, low_state, 1)

        # Send the motor commands to the robot
        udp.SetSend(cmd)
        udp.Send()
