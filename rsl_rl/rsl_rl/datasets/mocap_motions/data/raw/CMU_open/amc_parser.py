import numpy as np
import re
from scipy.spatial.transform import Rotation as R
from rsl_rl.datasets.motion_retarget import MotionRetarget, forward_kinematics
import isaacgym
import torch
import torch
import torch.nn.functional as F
import os
##############HELPER#############
def euler_to_angular_velocity(euler_angles, dt, sequence='XYZ'):
    """
    Convert the change in Euler angles to angular velocity.
    
    Args:
    euler_angles: Tensor of shape (T, 3) where T is the number of frames.
    dt: Time difference between frames.
    sequence: The sequence of rotations for Euler angles (default is 'XYZ').

    Returns:
    angular_velocity: Tensor of shape (T-1, 3) containing angular velocities.
    """
    # Compute the difference in Euler angles between consecutive frames
    delta_euler = (euler_angles[1:] - euler_angles[:-1]) / dt
    
    # Initialize angular velocity tensor
    T = euler_angles.shape[0]
    angular_velocity = torch.zeros((T-1, 3))
    
    for i in range(T-1):
        # Extract the Euler angles for the current frame
        roll, pitch, yaw = euler_angles[i]
        
        if sequence == 'XYZ':
            # Compute the transformation matrix from Euler angle rates to angular velocity
            transformation_matrix = torch.tensor([
                [1, 0, -torch.sin(pitch)],
                [0, torch.cos(roll), torch.cos(pitch) * torch.sin(roll)],
                [0, -torch.sin(roll), torch.cos(pitch) * torch.cos(roll)]
            ], device='cuda:0')
        elif sequence == 'ZYX':
            # You can add other sequences like ZYX, YXZ, etc.
            # The transformation matrix would change accordingly.
            pass
        
        # Compute angular velocity
        angular_velocity[i] = torch.matmul(transformation_matrix, delta_euler[i])
    return angular_velocity


def quaternion_to_euler(quaternions, sequence='XYZ'):
    """
    Convert a tensor of quaternions to Euler angles.
    
    Args:
    quaternions: Tensor of shape (T, 4), where T is the number of frames.
    sequence: The sequence of rotations for Euler angles (default is 'XYZ').
    
    Returns:
    euler_angles: Tensor of shape (T, 3) containing Euler angles.
    """
    # Ensure the quaternions are normalized
    quaternions = F.normalize(quaternions, p=2, dim=1)

    # Extract individual components of the quaternions
    qw, qx, qy, qz = 3, 0, 1, 2

    w, x, y, z = quaternions[:, qw], quaternions[:, qx], quaternions[:, qy], quaternions[:, qz]
    
    if sequence == 'XYZ':
        # Compute Euler angles based on XYZ rotation sequence
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clamp(t2, -1.0, 1.0)  # Clamp to avoid numerical issues
        pitch_y = torch.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)

        euler_angles = torch.stack((roll_x, pitch_y, yaw_z), dim=1)
    else:
        # Handle other sequences like ZYX, YXZ, etc.
        raise NotImplementedError(f"Euler sequence {sequence} is not implemented")
    
    return euler_angles

##################################

def parse_asf(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    bone_data = {}
    read_hierarchy = False
    for line in lines:
        if ':hierarchy' in line:
            read_hierarchy = True
        if read_hierarchy:
            if 'begin' in line or 'end' in line or ':hierarchy' in line:
                continue
            parts = line.split()
            if len(parts) > 1:
                parent = parts[0]
                children = parts[1:]
                for child in children:
                    bone_data[child] = parent

    return bone_data

def parse_amc(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    frames = []
    frame_data = None

    for line in lines:
        if line.strip().isdigit():
            if frame_data is not None:
                frames.append(frame_data)
            frame_data = {}
        else:
            parts = line.strip().split()
            if len(parts) > 1:
                frame_data[parts[0]] = list(map(float, parts[1:]))

    if frame_data is not None:
        frames.append(frame_data)

    return frames

def transform_coordinates(pos):
    x, y, z = pos
    return np.array([z, x, y])

def extract_joint_angles(data, bone_hierarchy):
    joint_order = ['lfemur', 'ltibia', 'lfoot', 'rfemur', 'rtibia', 'rfoot']
    angles = []
    for joint in joint_order:
        if joint == 'lfemur' or joint == 'rfemur':
            angles.extend([data[joint][0], data[joint][1], data[joint][2]])
        elif joint == 'ltibia' or joint == 'rtibia':
            angles.append(data[joint][0])
        elif joint == 'lfoot' or joint == 'rfoot':
            angles.extend([data[joint][0], data[joint][1]])
    return np.array(angles)

def convert_to_quaternion(rot):
    r = R.from_euler('xyz', rot, degrees=True)
    q = r.as_quat()
    return q  # returns (x, y, z, w)

def amc_to_numpy(asf_filename, amc_filename):
    bone_hierarchy = parse_asf(asf_filename)
    frames = parse_amc(amc_filename)

    T = len(frames)
    euler_ver = np.zeros((T, 18))
    quat_ver = np.zeros((T, 19))

    for i, frame in enumerate(frames):
        root_pos = frame['root'][:3]
        root_rot = frame['root'][3:]
        
        transformed_pos = transform_coordinates(root_pos)
        transformed_rot = transform_coordinates(root_rot)
        euler = transform_coordinates(transformed_rot)* np.pi / 180
        quaternion = convert_to_quaternion(transformed_rot)

        euler_ver[i, :3] = transformed_pos
        euler_ver[i, 3:6] = euler 
        quat_ver[i, :3] = transformed_pos
        quat_ver[i, 3:7] = quaternion

        joint_angles = extract_joint_angles(frame, bone_hierarchy)* np.pi / 180
        euler_ver[i, 6:] = joint_angles 
        quat_ver[i, 7:] = joint_angles

    return euler_ver, quat_ver

asf_filename = '07.asf'
amc_filename = 'walk_1.amc'
xml_filename = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/cmu.xml'
FOOT_NAME = ['L_AnkleRoll_Joint', 'R_AnkleRoll_Joint']
dt = 1/120

model_processor = MotionRetarget(xml_filename, xml_filename)
q_euler, q_quat = amc_to_numpy(asf_filename, amc_filename)
q_euler = torch.tensor(q_euler, device='cuda:0', dtype=torch.float)
q_quat = torch.tensor(q_quat, device='cuda:0', dtype=torch.float)
reference_length = q_euler.shape[0]
assert q_euler.shape[1] == 18, f"q_euler shape is {q_euler.shape}"
assert q_quat.shape[1] == 19, f"q_quat shape is {q_quat.shape}"
global_joint_pos, global_joint_quat, local_joint_pos, local_joint_quat = forward_kinematics(model_processor.source, q_quat, model_processor.JOINT_MAPPING)
L_foot_pos = local_joint_pos[FOOT_NAME[0]]
R_foot_pos = local_joint_pos[FOOT_NAME[1]]
L_foot_rpy = quaternion_to_euler(local_joint_quat[FOOT_NAME[0]])
R_foot_rpy = quaternion_to_euler(local_joint_quat[FOOT_NAME[1]])

q_pose = q_euler[:, 6:]
q_prev, q_vel = torch.zeros_like(q_pose), torch.zeros_like(q_pose)
q_prev[1:, :] = q_pose[:-1, :]
q_vel[1:, :] = (q_pose[1:, :] - q_prev[1:, :]) / dt
q_vel[0, :] = 0.

base_pos = q_euler[:, :3]
base_euler = q_euler[:, 3:6]
base_linvel, base_angvel = torch.zeros_like(base_pos), torch.zeros_like(base_euler)
base_linvel[1:, :], base_angvel[1:, :] = (base_pos[1:, :] - base_pos[:-1, :])/dt, euler_to_angular_velocity(base_euler, dt)
base_linvel[0,:], base_angvel[0,:] = 0.,0. 

time_stamp = torch.tensor([i*dt for i in range(reference_length)], dtype=torch.float, device='cuda:0')

reference_tensor = torch.cat((time_stamp.unsqueeze(1), base_pos, base_euler, base_linvel, base_angvel, q_pose, q_vel, L_foot_pos, L_foot_rpy,  R_foot_pos, R_foot_rpy), dim=-1)


# Save numpy array to a file
file_name = os.path.splitext(os.path.basename(amc_filename))[0]+'.txt'
np.savetxt(file_name, reference_tensor.detach().cpu().numpy(), fmt='%f', delimiter='    ')

print(f"Conversion complete. Numpy array saved as {file_name}.")


