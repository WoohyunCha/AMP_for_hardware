import re, json
import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
import isaacgym
import torch
import torch.nn.functional as F
import os
from lxml import etree
import numpy as np

from isaacgym.torch_utils import *
import glob
import torch.optim as optim




'''
AMC/ASF PARSER + RETARGET
RUN THIS SCRIPT AFTER RUNNING asf_parser.py!!!!!
This script converts a given pair of ASF/AMC files into XML/JSON files, then retargets the JSON file according to the TARGET XML file.
ASF file contains information about the skeleton model of the MoCap subject.
AMC file contains a trajectory of joint angles.
The converted XML file contains information extracted from the ASF file. Note that only the geometric information are valid.
The converted JSON file contains a trajectory of certain geometric values (ex. joint angles, joint velocities, foot positions, etc)
The TARGET XML file is the description of the model to which you wish to retarget the ASF/AMC motion.
In order to change the contents of the retarget output, find ### FROM HEREON, CONSTRUCT THE RETARGET TRAJECTORY!!!###

DO NOT FORGET TO CHECK THE FOLLOWINGS:
1. Specify the path to the ASF/AMC files, the converted XML file, the path to where you wish to save the converted JSON files, the path to the TARGET XML file.
2. Specify the frame rate of the MoCap data.
3. Specify the target model's XML file.
4. The XML file and the TARGET XML file should share joint names. The joint axes can be different but the hierarchy of the joint names should be the same.
    YOU MUST comment out the upper body parts of the converted XML file and change the names of the joints so that they match those of the TARGET XML file.
5. Specify target model's init pos (TOCABI_INIT_POS)
6. Specify MODEL_DOF, which is the DOF of the source XML file
7. The result of retarget cannot include base link information in global frame coordinates, as these are hard to retarget
8. Specify the names of the joints closest to the feet, in the list "FOOT_NAME". 
9. Delete the first few lines (before the frame starts) of the AMC.

COMMON ERROR
1. Optimization happens once more than expected -> Make sure to erase the MAIN FUNCTION when using this script by importing it.
'''

# ASF_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/91/asf/91.asf' # Path to your skeleton ASF file
# AMC_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/91/amc/straight_walk.amc'
# XML_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/91/xml/91.xml' # Path of the converted source XML file

# ASF_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/07/asf/07.asf' # Path to your skeleton ASF file
# AMC_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/07/amc/walk_1.amc'
# XML_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/07/xml/07.xml' # Path of the converted source XML file

ASF_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/69/asf/69.asf' # Path to your skeleton ASF file
AMC_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/69/amc/walk_forward_01.amc'
XML_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/69/xml/69.xml' # Path of the converted source XML file
TARGET_XML_FILE = '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi.xml'
TXT_FILE = os.path.join('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open',os.path.splitext(os.path.basename(ASF_FILE))[0], 'txt', os.path.splitext(os.path.basename(AMC_FILE))[0]+'.txt')
JSON_FILE = os.path.join('/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/motions_json/cmu/', os.path.splitext(os.path.basename(ASF_FILE))[0], os.path.splitext(os.path.basename(ASF_FILE))[0] + '_' + os.path.splitext(os.path.basename(AMC_FILE))[0]+'.json')
RETARGET_FILE = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/retarget_motions/retarget_reference_data.txt'
HZ = 120
dt = 1/HZ # Frame rate of ASF/AMC
MODEL_DOF = 12
TOCABI_INIT_POS = [
    0.0,
    0.0,
    -0.28,
    0.6,
    -0.32,
    0.0,
    0.0,
    0.0,
    -0.28,
    0.6,
    -0.32,
    0.0,
]
FOOT_NAME = ['L_AnkleRoll_Joint', 'R_AnkleRoll_Joint']



#### MACRO definitions. MODIFY WITH CARE####
VIRTUAL_JOINT_DIM = 7
FRAME_TIME = 0
POS_SIZE = 3
JOINT_POS_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
ROOT_POS_SIZE = 1 # base height change
ROOT_ROT_SIZE = 4 # projected gravity

OBSERVATION_DIM = ROOT_POS_SIZE + ROOT_ROT_SIZE + LINEAR_VEL_SIZE + ANGULAR_VEL_SIZE\
        + JOINT_POS_SIZE + JOINT_VEL_SIZE + 2*(POS_SIZE)

ROOT_POS_START_IDX = 0
ROOT_POS_END_IDX = ROOT_POS_START_IDX + ROOT_POS_SIZE

ROOT_ROT_START_IDX = ROOT_POS_END_IDX
ROOT_ROT_END_IDX = ROOT_ROT_START_IDX + ROOT_ROT_SIZE

LINEAR_VEL_START_IDX = ROOT_ROT_END_IDX
LINEAR_VEL_END_IDX = LINEAR_VEL_START_IDX + LINEAR_VEL_SIZE

ANGULAR_VEL_START_IDX = LINEAR_VEL_END_IDX
ANGULAR_VEL_END_IDX = ANGULAR_VEL_START_IDX + ANGULAR_VEL_SIZE

JOINT_POSE_START_IDX = ANGULAR_VEL_END_IDX
JOINT_POSE_END_IDX = JOINT_POSE_START_IDX + JOINT_POS_SIZE

JOINT_VEL_START_IDX = JOINT_POSE_END_IDX
JOINT_VEL_END_IDX = JOINT_VEL_START_IDX + JOINT_VEL_SIZE

FOOT_POS_START_IDX = JOINT_VEL_END_IDX
FOOT_POS_END_IDX = FOOT_POS_START_IDX + 2*POS_SIZE

TOCABI_INIT_POS_TORCH = torch.tensor(TOCABI_INIT_POS, dtype=torch.float32, device='cuda:0').reshape(1,-1)



def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def sanitize(vec3):
    """ Swaps axes so that the Mocap coordinates fit Mujoco's coordinate system. """
    if len(vec3) != 3:
        return vec3
    return [round(vec3[2], 3), round(vec3[0], 3), round(vec3[1], 3)]


def rot_euler(v, xyz):
    """ Rotate vector v (or array of vectors) by the euler angles xyz. """
    # https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    for theta, axis in zip(xyz, np.eye(3)):
        v = np.dot(np.array(v), expm(np.cross(np.eye(3), axis*-theta)))
    return v

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

    
def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles (ZYX) to a rotation matrix. """
    B = euler_angles.shape[0]
    c = torch.cos(euler_angles)
    s = torch.sin(euler_angles)

    # Preallocate rotation matrix
    R = torch.zeros((B, 3, 3), dtype=torch.float32, device=euler_angles.device)

    # Fill in the entries
    R[:, 0, 0] = c[:, 1] * c[:, 0]
    R[:, 0, 1] = c[:, 1] * s[:, 0]
    R[:, 0, 2] = -s[:, 1]
    R[:, 1, 0] = s[:, 2] * s[:, 1] * c[:, 0] - c[:, 2] * s[:, 0]
    R[:, 1, 1] = s[:, 2] * s[:, 1] * s[:, 0] + c[:, 2] * c[:, 0]
    R[:, 1, 2] = s[:, 2] * c[:, 1]
    R[:, 2, 0] = c[:, 2] * s[:, 1] * c[:, 0] + s[:, 2] * s[:, 0]
    R[:, 2, 1] = c[:, 2] * s[:, 1] * s[:, 0] - s[:, 2] * c[:, 0]
    R[:, 2, 2] = c[:, 2] * c[:, 1]

    return R

def transform_vector(euler_angles, vectors):
    """ Transform vectors from frame A to B using a batch of Euler angles.
    
    Args:
    - euler_angles (numpy.array): Array of shape [B, 3] containing Euler angles.
    - vectors (numpy.array): Array of shape [B, 3] containing vectors in frame A.
    
    Returns:
    - transformed_vectors (numpy.array): Array of shape [B, 3] containing vectors in frame B.
    """
    # Convert inputs to torch tensors
    euler_angles = torch.tensor(euler_angles, dtype=torch.float32)
    vectors = torch.tensor(vectors, dtype=torch.float32)

    # Get rotation matrices from Euler angles
    R = euler_to_rotation_matrix(euler_angles)

    # Transform vectors
    transformed_vectors = torch.bmm(R, vectors.unsqueeze(-1)).squeeze(-1)

    # Convert back to numpy arrays
    return transformed_vectors.numpy()

def write_tensor_to_txt(tensor, filename):
    """
    Writes a torch tensor to a text file, with each tensor row on a new line.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (N, d) to write to the file.
    filename (str): Path to the output text file.
    """
    # Ensure the tensor is on CPU and convert it to numpy for easier handling
    data = tensor.cpu().numpy()
    
    # Open the file and write the tensor data
    with open(filename, 'w') as f:
        for row in data:
            # Create a string for each row with elements separated by spaces
            row_string = ' '.join(map(str, row))
            # Write the row string to the file and add a newline character
            f.write(row_string + '\n')

def forward_kinematics(model: dict, q: torch.Tensor, joint_map: dict, device = 'cuda:0', requires_grad = False):
    '''
    Computes the forward kinematics for a trajectory, given model = {bodies: dict, joints: dict}, q = torch.Tensor((traj_length, num_joints+7))
    joint_map is a dict that maps the joint names to their indices in q. For example, virtual joint = 0, hip yaw joint = 7, ...
    '''
    VIRTUAL_JOINT_DIM = 7 # pos and quat
    TRAJECTORY_LENGTH = q.shape[0]

    bodies, joints = model['bodies'], model['joints']
    num_joints = q.shape[-1]-VIRTUAL_JOINT_DIM # q includes virtual joint. The rotation part is in quat.
    # The position and orientation of the frame attached to the joint. This will be the body frame for the child link.
    global_joint_pos = {}
    global_body_pos = {}
    global_joint_rot = {}
    global_body_rot = {}
    local_body_pos = {}
    local_body_rot = {}
    local_joint_pos = {}
    local_joint_rot = {}

    global_base_rot = None

    for body_name, body in bodies.items():
        parent_name = body['parent']
        joint_name = body['joint']
        body_info = body['info']
        if joint_name is not None:
            joint_info = joints[joint_name]

        if body_name == "base_link": # Let's assume there is no virtual joint offset from base link frame origin
            virtual_joint_value = q[:, joint_map[joint_name]:joint_map[joint_name]+VIRTUAL_JOINT_DIM]
            global_joint_pos[joint_name] = virtual_joint_value[:, 0:3]
            global_joint_rot[joint_name] = virtual_joint_value[:, 3:VIRTUAL_JOINT_DIM]
            global_body_pos[body_name] = global_joint_pos[joint_name]
            global_body_rot[body_name] = global_joint_rot[joint_name]
            global_base_pos = global_joint_pos[joint_name]
            global_base_rot = global_joint_rot[joint_name]
            local_joint_pos[joint_name] = torch.zeros_like(global_joint_pos[joint_name])
            local_joint_rot[joint_name] = torch.zeros_like(global_joint_rot[joint_name])
            local_body_pos[body_name] = torch.zeros_like(global_base_pos)
            local_body_rot[body_name] = torch.zeros_like(global_base_rot)

            local_joint_rot[joint_name][:, -1] = 1. # Quaternion zero.
        elif joint_name in joint_map:
            q_ = q[:, joint_map[joint_name]]
            body_offset = to_torch(body_info['position']).tile(TRAJECTORY_LENGTH, 1)
            joint_offset = to_torch(joint_info['position']).tile(TRAJECTORY_LENGTH, 1)
            body_rot_offset = to_torch(body_info['quat']).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
            joint_rot_offset = to_torch(joint_info['quat']).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
            global_body_rot[body_name] = quat_mul(quat_mul(global_body_rot[parent_name], quat_from_angle_axis(q_, to_torch(joint_info['axis']))), body_rot_offset)
            global_body_pos[body_name] = global_body_pos[parent_name] + quat_rotate(global_body_rot[parent_name], body_offset)
            global_joint_pos[joint_name] = global_body_pos[body_name] + quat_rotate(global_body_rot[body_name], joint_offset)
            global_joint_rot[joint_name] = quat_mul(global_body_rot[body_name], joint_rot_offset)

            local_body_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_body_pos[body_name] - global_base_pos))
            local_body_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_body_rot[body_name])
            local_joint_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_joint_pos[joint_name] - global_base_pos))
            local_joint_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_joint_rot[joint_name])

        elif joint_name is None:
            # body is welded to parent
            body_offset = to_torch(body_info['position']).tile(TRAJECTORY_LENGTH, 1)
            body_rot_offset = to_torch(body_info['quat']).tile(TRAJECTORY_LENGTH, 1)[:, [1,2,3,0]]
            global_body_rot[body_name] = quat_mul(global_body_rot[parent_name], body_rot_offset)
            global_body_pos[body_name] = global_body_pos[parent_name] + quat_rotate(global_body_rot[parent_name], body_offset)
            local_body_pos[joint_name] = quat_rotate_inverse(global_base_rot, (global_body_pos[body_name] - global_base_pos))
            local_body_rot[joint_name] = quat_mul(quat_conjugate(global_base_rot), global_body_rot[body_name])
        else:
            # upper body
            pass
    return global_joint_pos, global_joint_rot, local_joint_pos, local_joint_rot


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

def denoise(data, window_length=5, polyorder=2):
    return savgol_filter(data, window_length=window_length, polyorder=polyorder, axis=0)

def process_data(traj: np.ndarray):# Process raw data from Mujoco
    traj = denoise(traj, window_length=20, polyorder=2)
    time = traj[:,0]
    pelv_pos = traj[:, 1:4]
    pelv_rpy = traj[:, 4:7] # roll, pitch, yaw order
    pelv_vel = traj[:, 7:13]
    q_pos = traj[:, 13:13+JOINT_POS_SIZE]
    start = 13+MODEL_DOF
    end = start + JOINT_VEL_SIZE
    q_vel = traj[:, start:end]
    start += MODEL_DOF
    end = start + 3
    L_foot_pos = traj[:, start:end]
    start = end
    end = start + 3
    L_foot_rpy = traj[:, start:end]
    start = end
    end = start + 3
    R_foot_pos = traj[:, start:end]
    start=end
    end = start + 3
    R_foot_rpy = traj[:, start:end]
    assert end == 1+3+3+6+MODEL_DOF+MODEL_DOF+3+3+3+3, f"process data, shape mismatch {end}"

    # torch versions of data
    pelv_pos_torch = to_torch(pelv_pos)
    pelv_vel_torch = to_torch(pelv_vel)
    q_pos_torch = to_torch(q_pos)
    q_vel_torch = to_torch(q_vel)
    L_foot_pos_torch = to_torch(L_foot_pos)
    L_foot_rpy_torch = to_torch(L_foot_rpy)
    R_foot_pos_torch = to_torch(R_foot_pos)
    R_foot_rpy_torch = to_torch(R_foot_rpy)

    # Process raw data
    pelvis_yaw =pelv_rpy[:, 2]
    pelvis_pitch = pelv_rpy[:, 1]
    pelvis_roll = pelv_rpy[:, 0]
    pelvis_quat = quat_from_euler_xyz(to_torch(pelvis_roll), to_torch(pelvis_pitch), to_torch(pelvis_yaw)) # tensor
    base_pos_global_torch = pelv_pos_torch 

    # Create AMP observation
    base_height = base_pos_global_torch[:, 2]
    assert (~torch.isfinite(base_height)).sum() == 0, "Found non finite element"
    assert (~torch.isfinite(pelvis_quat)).sum() == 0, "Found non finite element 0"
    # gravity_vector = to_torch(get_axis_params(-1., 2), device=self.device).repeat((pelvis_yaw.shape[0], 1))
    # projected_gravity = quat_rotate_inverse(pelvis_quat, gravity_vector)
    base_lin_vel = quat_rotate_inverse(pelvis_quat, pelv_vel_torch[:, :3]) 
    assert (~torch.isfinite(base_lin_vel)).sum() == 0, "Found non finite element 1"
    base_ang_vel = quat_rotate_inverse(pelvis_quat, pelv_vel_torch[:, 3:])
    assert (~torch.isfinite(base_ang_vel)).sum() == 0, "Found non finite element 2"
    L_foot_pos_base_torch = quat_rotate_inverse(pelvis_quat, L_foot_pos_torch - base_pos_global_torch)
    assert (~torch.isfinite(L_foot_pos_base_torch)).sum() == 0, "Found non finite element 3"
    R_foot_pos_base_torch = quat_rotate_inverse(pelvis_quat, R_foot_pos_torch - base_pos_global_torch) 
    assert (~torch.isfinite(R_foot_pos_base_torch)).sum() == 0, "Found non finite element 4"
    L_foot_quat_global_torch = quat_from_euler_xyz(L_foot_rpy_torch[:, 0], L_foot_rpy_torch[:, 1], L_foot_rpy_torch[:, 2])
    assert (~torch.isfinite(L_foot_quat_global_torch)).sum() == 0, "Found non finite element 5"
    R_foot_quat_global_torch = quat_from_euler_xyz(R_foot_rpy_torch[:, 0], R_foot_rpy_torch[:, 1], R_foot_rpy_torch[:, 2])
    assert (~torch.isfinite(R_foot_quat_global_torch)).sum() == 0, "Found non finite element 6"
    L_foot_quat_base_torch = quat_mul(quat_conjugate(pelvis_quat), L_foot_quat_global_torch)
    assert (~torch.isfinite(L_foot_quat_base_torch)).sum() == 0, "Found non finite element 7"
    R_foot_quat_base_torch = quat_mul(quat_conjugate(pelvis_quat), R_foot_quat_global_torch)
    assert (~torch.isfinite(R_foot_quat_base_torch)).sum() == 0, "Found non finite element 8"
    # ret = torch.concat((pelvis_quat, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
    ret = torch.concat((base_height.unsqueeze(-1), pelvis_quat, base_lin_vel , base_ang_vel, q_pos_torch, q_vel_torch, L_foot_pos_base_torch, R_foot_pos_base_torch), dim=-1) 
    info = torch.concat((base_height.unsqueeze(-1), base_lin_vel, base_ang_vel, L_foot_quat_base_torch, R_foot_quat_base_torch), dim=-1)

    return ret, info

class AsfParser(object):
    def __init__(self):
        self.hierarchy = {}
        self.root = {}
        self.bones = {}

    def parse(self, file_name):
        """ Loads ASF Mocap file and parses the bone and joint hierarchy. """
        with open(file_name, 'r') as f:
            lines = list(f)
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#'):
                i += 1
                continue
            if line.startswith(':root'):
                while i < len(lines) - 1:
                    i += 1
                    line = lines[i].strip()
                    if line.startswith(':'):
                        break
                    elif line.startswith('position') or line.startswith('orientation'):
                        split = line.split()
                        self.root[split[0]] = [float(x) for x in split[1:]]
            elif line.startswith(':bonedata'):
                bone = {}
                while i < len(lines) - 1:
                    i += 1
                    line = lines[i].strip()
                    split = line.split()
                    if line.startswith(':'):
                        break
                    elif line.startswith('begin'):
                        bone = { 'id': 0 }
                    elif line.startswith('end'):
                        self.bones[bone['name']] = bone
                    elif line.startswith('direction') or line.startswith('axis'):
                        bone[split[0]] = [float(x) for x in split[1:] if isfloat(x)]
                    elif line.startswith('length'):
                        bone['length'] = float(split[1])
                    elif line.startswith('name'):
                        bone['name'] = split[1]
                    elif line.startswith('dof'):
                        bone['dof'] = split[1:]
                    elif line.startswith('id'):
                        bone['id'] = int(split[1])
                    elif line.startswith('limits'):
                        line = ' '.join(split[1:])
                        bone['limits'] = []
                        while i < len(lines) and re.search('\((.*?)\)', line):
                            bone['limits'].append([float(x) for x in re.findall('\((.*?)\)', line)[0].split()])
                            i += 1
                            line = lines[i].strip()
                        if len(bone['limits']) > 0:
                            i -= 1
            elif line.startswith(':hierarchy'):
                links = {}
                while i < len(lines) - 1:
                    i += 1
                    line = lines[i].strip()
                    split = line.split()
                    if line.startswith(':') or line.startswith('end'):
                        break
                    elif line.startswith('begin'):
                        pass
                    else:
                        if split[0] == 'root':
                            self.hierarchy['root'] = {
                                'name': 'root',
                                'children': {x: self.bones[x] for x in split[1:]}
                            }
                            for x in split[1:]:
                                links[x] = self.hierarchy['root']['children'][x]
                        else:
                            links[split[0]]['children'] = {x: self.bones[x] for x in split[1:]}
                            for x in split[1:]:
                                links[x] = links[split[0]]['children'][x]
            else:
                i += 1

    def save_json(self, file_name):
        """ Saves the bone hierarchy as JSON file. """
        return json.dump(self.hierarchy, open(file_name, 'w'), indent=4)

    def save_mujoco_xml(self, file_name):
        """ Saves the skeleton as Mujoco/Roboschool-compatible XML file. """
        xml = '''<mujoco model="humanoid">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="3" friction="0.8 0.1 0.1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003">
        <!-- <flags solverstat="enable" energy="enable"/>-->
    </option>
    <size nkey="5" nuser_geom="1"/>
    <visual>
        <map fogend="5" fogstart="3"/>
    </visual>
    <worldbody>
        '''

        def traverse(node, level=2):
            snippet = ''
            name = node['name']
            vector = np.array([0, 0, 0])
            if name == 'root':
                snippet += '<body name="torso" pos="0 0 10">\n'
                level += 1
            else:
                vector = np.array(node['direction']) * node['length'] / 15.
                snippet += '\t'*level + '<body name="{name}" pos="{v[0]} {v[1]} {v[2]}">\n'.format(name=name, v=sanitize(vector))
                level += 1
                # snippet += '\t'*level + ('<geom fromto="{vf[0]} {vf[1]} {vf[2]} {vt[0]} {vt[1]} {vt[2]}" ' +
                #                          'name="{name}" size="0.05" type="capsule" />\n').format(
                #                         **node, vf=sanitize(-vector), vt=[0,0,0])

            if 'axis' not in node:
                euler = np.zeros(3)
            else:
                euler = np.array(node['axis'])*np.pi/180.

            snippet += '\t'*level + ('<geom fromto="{vf[0]} {vf[1]} {vf[2]} {vt[0]} {vt[1]} {vt[2]}" ' +
                                     'name="{name}" size="0.05" type="capsule" />\n').format(
                                    **node, vf=sanitize(-vector), vt=[0,0,0])

            snippet += '\t'*level + ('<geom contype="0" pos="{v[0]} {v[1]} {v[2]}" name="vis_{name}" size="0.001" type="sphere" rgba=".2 .5 1 1" />\n').format(
                            **node, v=sanitize(-vector))

            if 'dof' in node:
                colors = {'x': '1 0 0', 'y': '0 1 0', 'z': '0 0 1'}
                dims = 'xyz'
                for i, dof in enumerate(node['dof']):
                    dim = dof[1]
                    dim_index = dims.index(dim)
                    axis = np.zeros(3)
                    axis[dim_index] = 1
                    axis = np.array(sanitize(rot_euler(axis, euler)))
                    snippet += '\t'*level + '<joint '
                    snippet += ('armature="0.1" damping="0.5" name="{name}_{dim}" axis="{axis[0]} {axis[1]} {axis[2]}" '
                                + 'pos="{v[0]} {v[1]} {v[2]}" stiffness="{stiff}" type="hinge" range="{range_l} {range_u}"').format(
                        name=name, dim=dim, axis=axis, v=sanitize(-vector),
                        range_l=node['limits'][i][0], range_u=node['limits'][i][1],
                        stiff=400./node['length']**4.)
                    # print('%s has stiffness %.3f' % (node['name'], 400./node['length']**4.))
                    snippet += ' />\n'
                    # snippet += '\t'*level + ('<geom pos="{v[0]} {v[1]} {v[2]}" name="vis_{name}" size="0.06" type="sphere" rgba=".2 .5 1 1" />\n').format(
                    #                     **node, v=sanitize(-vector))
                    # snippet += '\t'*level + '<geom fromto="{s[0]} {s[1]} {s[2]} {v[0]} {v[1]} {v[2]}" type="capsule" size="0.01" rgba="{color} 1" />\n'.format(s=sanitize(-vector), v=sanitize(rot_euler(axis, euler)*0.3-vector), color=colors[dim])

            if 'children' in node:
                snippet += '\n'.join([traverse(x, level) for x in node['children'].values()])

            level -= 1
            snippet += '\t'*level + '</body>\n'
            return snippet

        xml += traverse(self.hierarchy['root'])

        xml += '''\t</worldbody>
    <tendon>
    </tendon>
    <actuator><!-- this section is not supported, same constants in code -->'''
        for j in self.bones.values():
            if "dof" not in j:
                continue
            for dof in j["dof"]:
                dim = dof[1]
                xml += '\n' + '\t'*2 + '<motor gear="100" joint="{name}_{dim}" name="{name}_{dim}"/>'.format(**j, dim=dim)
        xml += '''
    </actuator>
</mujoco>
'''
        with open(file_name, 'w') as f:
            f.write(xml)




class MotionRetarget():

    JOINT_MAPPING = {
        'virtual_joint': 0,
        'L_HipYaw_Joint': 7, 
        'L_HipRoll_Joint': 8,
        'L_HipPitch_Joint': 9,
        'L_Knee_Joint': 10,
        'L_AnklePitch_Joint': 11,
        'L_AnkleRoll_Joint': 12,

        'R_HipYaw_Joint': 13, 
        'R_HipRoll_Joint': 14,
        'R_HipPitch_Joint': 15,
        'R_Knee_Joint': 16,
        'R_AnklePitch_Joint': 17,
        'R_AnkleRoll_Joint': 18,
    }

    def __init__(self, source_model_path: str, target_model_path: str):
        source = source_model_path
        target = target_model_path
        source_bodies, source_joints, source_edges = self.process_model(source)
        target_bodies, target_joints, target_edges = self.process_model(target)
        self.root_ratio = target_bodies['base_link']['info']['position'][-1] / source_bodies['base_link']['info']['position'][-1]
        assert self.root_ratio < 10 and self.root_ratio > 0.1, f"Root ratio is abnormally small, {self.root_ratio}"
        self.source = {
            'bodies': source_bodies,
            'joints': source_joints,
            'edges': source_edges
        }
        self.target = {
            'bodies': target_bodies,
            'joints': target_joints,
            'edges': target_edges
        }
        self.source_model_path = source_model_path
        self.target_model_path = target_model_path
        # print("SOURCE MODEL")
        # self.print_model(self.source)
        # print("TARGET MODEL")
        # self.print_model(self.target)
        

    def process_model(self, model_path: str):
        tree = etree.parse(model_path)
        root = tree.getroot()
        joints = {}
        bodies = {}
        edges = {}

    # Collect body and joint data
        for body in root.findall('.//body'):
            body_name = body.get('name')
            pos = np.array([float(num) for num in body.get('pos', '0 0 0').split()], dtype=float) # default value is 0, 0, 0
            quat = np.array([float(num) for num in body.get('quat', '1 0 0 0').split()], dtype=float) # default value is 0, 0, 0
            if body.find('inertial') is not None: # not a dummy
                inertial = body.find('inertial')
                mass = np.array([float(num) for num in inertial.get('mass', '0.').split()], dtype=float) # default value is 0, 0, 0
                com = np.array([float(num) for num in inertial.get('pos', '0 0 0').split()], dtype=float) # default value is 0, 0, 0
                inertia = np.array([float(num) for num in inertial.get('fullinertia', '0 0 0 0 0 0').split()], dtype=float) # default value is 0, 0, 0
            else:
                mass = None
                com = None
                inertia = None

            body_info = {
                'position': pos,
                'quat': quat,
                'mass': mass,
                'inertia': inertia,
                'com': com
            }

            parent = body.getparent()
            parent_name = parent.get('name')
            if parent_name is None:
                parent_name = 'world'
            
            if body.find('joint') is not None:
                for i, joint in enumerate(body.findall('joint')):
                    joint_name = joint.get('name')
                    joint_pos = np.array([float(num) for num in joint.get('pos', '0 0 0').split()], dtype=float)
                    joint_axis = np.array([float(num) for num in joint.get('axis', '0 0 1').split()], dtype=float)
                    joint_quat = np.array([float(num) for num in joint.get('quat', '1 0 0 0').split()], dtype=float)
                    joint_type = joint.get('type', 'hinge')
                    joint_range = np.array([float(num) for num in joint.get('range', '-3.14 3.14').split()], dtype=float)
                    joint_info = {
                        'parent' : parent_name,
                        'child' : body_name,
                        'position' : joint_pos,
                        'axis' : joint_axis,
                        'type' : joint_type,
                        'range' : joint_range,
                        'quat' : joint_quat
                    }
                    joints[joint_name] = joint_info
                    # process link length between joint_name and parent joint
                    edges[joint_name] = np.linalg.norm(body_info['position'], ord=2, axis=-1)
            else: # fixed link
                joint_name = None             
            bodies[body_name] = {'parent': parent_name, 'joint': joint_name, 'info' : body_info}    
            m = {'bodies': bodies, 'joints': joints}

        return bodies, joints, edges

    def print_model(self, model):
        bodies, joints, edges = model['bodies'], model['joints'], model['edges']
        print("PRINTING MODEL")
        print("---------------INFO----------------")
        for body_name, val in bodies.items():
            print("Link name : ", body_name)
            print(val)
            print("Joint name : ", val['joint'])
            print(joints[val['joint']]) if val['joint'] is not None else print("None")
            print("----------------------")
        print("-------------Link length-----------")
        for joint_name, val in edges.items():
            if joints[joint_name]['child'] != "base_link":
                print(f"{joint_name} and {bodies[joints[joint_name]['parent']]['joint']} : {val}")

    def retarget_tensor(self, reference: torch.Tensor, iterations :int,  play=False) -> torch.Tensor: # outputs retargetted reference
        length_ratios = {}
        reference_length = reference.shape[0]
        observation_dim = reference.shape[-1]
        for key, val in self.target['edges'].items():
            if key in self.JOINT_MAPPING:
                if self.source['edges'][key] == 0:
                    length_ratios[key] = 0.
                else:
                    length_ratios[key] = val/self.source['edges'][key]
        source_global_joint_pos = {}
        source_global_joint_quat = {}
        target_global_joint_pos = {}
        target_local_joint_pos = {}
        target_global_joint_quat = {}
        target_local_joint_quat = {}
        relative_position_vectors = {}
        source_bodies, source_joints= self.source['bodies'], self.source['joints']
        target_bodies, target_joints = self.target['bodies'], self.target['joints']

        start = 0
        end = start+ROOT_POS_SIZE
        # source_global_root_pos = torch.cat((torch.zeros_like(reference[:, start:end]), torch.zeros_like(reference[:, start:end]), reference[:, start:end]), dim=-1)
        source_global_root_pos = torch.cat((torch.zeros_like(reference[:, 0:1]), torch.zeros_like(reference[:, 0:1]), torch.zeros_like(reference[:, 0:1])), dim=-1)
        start = end
        end = start+ROOT_ROT_SIZE
        source_global_root_quat = reference[:, start:end]
        start = end
        end = start+LINEAR_VEL_SIZE
        source_local_base_lin_vel = reference[:, start:end]
        start = end
        end = start+ANGULAR_VEL_SIZE
        source_local_base_ang_vel = reference[:, start:end]
        start = end
        end = start+JOINT_POS_SIZE
        source_qpos = reference[:, start:end]
        start = end
        end = start+JOINT_VEL_SIZE
        source_qvel = reference[:, start:end]
        start = end
        end = start+POS_SIZE
        source_local_Lfoot_pos = reference[:, start:end]
        start = end
        end = start+POS_SIZE
        source_local_Rfoot_pos = reference[:, start:end]
        assert end == reference.shape[-1], f"Retarget, observation shape do not match. Must be {reference.shape[-1]} but is {end}"
        
        source_q = torch.concat((source_global_root_pos, source_global_root_quat, reference[:, JOINT_POSE_START_IDX:JOINT_POSE_END_IDX]), dim=-1)
        assert source_q.shape == (reference_length, 7+12)
        source_global_joint_pos, source_global_joint_quat, source_local_joint_pos, source_local_joint_quat = forward_kinematics(self.source, source_q, self.JOINT_MAPPING)
        for body_name, body in source_bodies.items():

            parent_name = body['parent']
            joint_name = body['joint']
            body_info = body['info']

            if joint_name in self.JOINT_MAPPING and body_name != "base_link":
                if 'HipYaw_Joint' in joint_name:
                    relative_position_vectors[joint_name] = to_torch(length_ratios[joint_name]).tile((reference_length, 1)) * \
                        (source_global_joint_pos[joint_name] - source_global_joint_pos['virtual_joint']) # scaled relative position vector from parent joint to current joint in global frame                    
                else:
                    relative_position_vectors[joint_name] = to_torch(length_ratios[joint_name]).tile((reference_length, 1)) * \
                        (source_global_joint_pos[joint_name] - source_global_joint_pos[source_bodies[parent_name]['joint']]) # scaled relative position vector from parent joint to current joint in global frame

        # 3. Starting from base link, sum up the relative position vectors and transform back to base frame coordinates              
        joint_pos_global_retarget = torch.zeros((reference_length, 3 * 13), dtype=torch.float, device='cuda:0')
        joint_pos_local_retarget = torch.zeros((reference_length, 3 * 12), dtype=torch.float, device='cuda:0')
        joint_quat_global_retarget = torch.zeros((reference_length, 4*13), dtype=torch.float, device='cuda:0')
        joint_quat_local_retarget = torch.zeros((reference_length, 4*12), dtype=torch.float, device='cuda:0')
        for body_name, body in source_bodies.items():
            parent_name = body['parent']
            joint_name = body['joint']
            body_info = body['info']
            base_pos = source_global_joint_pos[source_bodies['base_link']['joint']]
            base_quat = source_global_joint_quat[source_bodies['base_link']['joint']]

            if body_name == "base_link":
                target_global_joint_pos[joint_name] = base_pos
                target_global_joint_quat[joint_name] = base_quat
                joint_pos_global_retarget[:, 0:3] = base_pos
            elif joint_name in self.JOINT_MAPPING and body_name != "base_link":
                parent_joint_name = source_bodies[parent_name]['joint']
                if 'HipYaw_Joint' in joint_name:
                    parent_joint_name = 'virtual_joint'
                target_global_joint_pos[joint_name] = target_global_joint_pos[parent_joint_name] + relative_position_vectors[joint_name]
                target_global_joint_quat[joint_name] = source_global_joint_quat[joint_name]
                target_local_joint_pos[joint_name] = quat_rotate_inverse(base_quat, (target_global_joint_pos[joint_name] - base_pos)) # relative position of joint from base in base frame coordinates
                target_local_joint_quat[joint_name] = quat_mul(quat_conjugate(base_quat), target_global_joint_quat[joint_name]) # quat of joint frame in base frame coordinates. Only the foot orientation will be a constraint for the IK
                joint_pos_global_retarget[:, 3*(self.JOINT_MAPPING[joint_name]-6):3*(self.JOINT_MAPPING[joint_name]-5)] = target_global_joint_pos[joint_name]
                joint_pos_local_retarget[:, 3*(self.JOINT_MAPPING[joint_name]-7):3*(self.JOINT_MAPPING[joint_name]-6)] = target_local_joint_pos[joint_name]
                joint_quat_global_retarget[:, 4*(self.JOINT_MAPPING[joint_name]-6):4*(self.JOINT_MAPPING[joint_name]-5)] = target_global_joint_quat[joint_name]
                joint_quat_local_retarget[:, 4*(self.JOINT_MAPPING[joint_name]-7):4*(self.JOINT_MAPPING[joint_name]-6)] = target_local_joint_quat[joint_name]

        # # 4. Solve numerical IK from joint positions, foot positions and orientation
        weight_joint_pos, weight_foot_quat, weight_joint_pose, weight_hip_pose, weight_joint_vel = 1., 0., .5, 0., .1
        # weight_joint_pos, weight_foot_quat, weight_joint_pose, weight_hip_pose, weight_joint_vel = 1.2, 0.1, .1, 0.1, .1 For CMU refernece
        weight_norm = (weight_joint_pos+weight_foot_quat+weight_joint_pose+weight_hip_pose)
        weight_joint_pos /= weight_norm
        weight_foot_quat /= weight_norm
        weight_joint_pose/= weight_norm
        weight_hip_pose/= weight_norm
        weight_joint_vel /= weight_norm

        def cost_function(x):
            # x = [q for each joints] -> shape of 12
            # Initialize x as [0, reference[:, ROOT_ROT_START_IDX:ROOT_ROT_END_IDX], reference[:, JOINT_POSE_START_IDX:JOINT_POSE_END_IDX]]
            # joint_pos_global_retarget = [base_pos, joint_pos] in global frame coordinates
            # joint_pos_local_retarget = [joint_pos] in base local frame coordinates

            temp = torch.zeros((x.shape[0], 7), requires_grad=False, device='cuda:0')
            temp[:, -1] = 1.
            _, x_global_joint_quat, x_local_joint_pos, x_local_joint_quat = forward_kinematics(self.target, torch.cat((temp, x), dim=-1), self.JOINT_MAPPING)
            x_local_joint_pos_tensor = torch.zeros_like(joint_pos_local_retarget)
            x_local_joint_quat_tensor = torch.zeros_like(joint_quat_local_retarget)
            for joint_name, joint_index in self.JOINT_MAPPING.items():
                if key == 'virtual_joint':
                    pass
                else:
                    index = joint_index-7
                    x_local_joint_pos_tensor[:, 3*index:3*(index+1)] = x_local_joint_pos[joint_name]
                    x_local_joint_quat_tensor[:, 4*index:4*(index+1)] = x_local_joint_quat[joint_name]
            assert x_local_joint_pos_tensor.shape == (reference_length, 36), f"local joint pos tensor shape is {x_local_joint_pos_tensor.shape}"

            mask = [0,1,2,3,6,7,8,9]
            hip_mas = [0,1, 6,7]
            # mask_ref = [i+JOINT_POSE_START_IDX for i in mask]
            # joint_pose_cost = torch.nn.MSELoss()(x[:, mask], TOCABI_INIT_POS_TORCH[:, mask]) # For CMU reference
            joint_pose_cost = torch.nn.MSELoss()(x, source_qpos)
            joint_pos_cost = torch.nn.MSELoss()(x_local_joint_pos_tensor, joint_pos_local_retarget)
            joint_vel_cost = torch.nn.MSELoss()(x[1:], x[:-1])
            hip_pose_cost = torch.nn.MSELoss()(x[:, hip_mas], TOCABI_INIT_POS_TORCH[:, hip_mas])
# 
            foot_quat_local_target_inverse = reference[:, ROOT_ROT_START_IDX:ROOT_ROT_END_IDX] # We want the foot to be flat to the ground. b_R_foot = b_R_g = g_R_b^-1
            L_ankle_roll_joint = FOOT_NAME[0]
            R_ankle_roll_joint = FOOT_NAME[1]

            x_Lfoot_quat = x_global_joint_quat[L_ankle_roll_joint]
            x_Rfoot_quat = x_global_joint_quat[R_ankle_roll_joint]
            zero_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float, device='cuda:0', requires_grad=False).tile(reference_length,1)
            # foot_quat_cost = (distance_between_quats(zero_quat, x_Lfoot_quat, neglect_axis=2).pow(2) \
            #                   + distance_between_quats(zero_quat, x_Rfoot_quat, neglect_axis=2).pow(2)).mean()
            cosine_loss = torch.nn.CosineSimilarity(dim=-1)
            foot_quat_cost = (1 - cosine_loss(zero_quat, x_Lfoot_quat)).mean() + (1 - cosine_loss(zero_quat, x_Rfoot_quat)).mean()

            total_cost = weight_joint_pos*joint_pos_cost + weight_joint_pose*joint_pose_cost + weight_foot_quat*foot_quat_cost + weight_hip_pose*hip_pose_cost + weight_joint_vel*joint_vel_cost
            return torch.sum(total_cost), joint_pos_cost, joint_pose_cost, (1 - cosine_loss(zero_quat, x_Lfoot_quat)).mean(), (1 - cosine_loss(zero_quat, x_Rfoot_quat)).mean()
        
        n_iterations = iterations
        if play:
            n_iterations = 0
        # q_opt = torch.zeros((reference_length, 12), dtype=torch.float32, requires_grad=True, device='cuda:0')
        # q_opt = reference[:, JOINT_POSE_START_IDX:JOINT_POSE_END_IDX].clone().detach().requires_grad_(True)
        q_opt = (TOCABI_INIT_POS_TORCH.tile((reference_length, 1))).clone().detach()
        q_opt += 0.1*torch.rand_like(q_opt)
        q_opt = q_opt.requires_grad_(True)
        assert q_opt.shape == (reference_length, 12)
        optimizer = optim.Adam([q_opt], lr=1e-3)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
        print("SOLVING IK...")
        for i in range(n_iterations):
            optimizer.zero_grad()
            cost = cost_function(q_opt)[0]
            cost.backward()
            torch.nn.utils.clip_grad_norm_(q_opt, max_norm=1.0)
            optimizer.step()
            scheduler.step(cost)
            if i%100 == 0:
                print(f"Iteration {i}, Cost: {cost.item()}")

            # Optionally, you can add some stopping criterion based on the change in cost or other conditions
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print("Stopping criterion met")
                break
            assert ~torch.any(~torch.isfinite(q_opt)), f"Cannot solve IK! Cost : {cost_function(q_opt)}"
        print("IK SOLVED!")

        temp = torch.zeros((q_opt.shape[0], 7), requires_grad=False, device='cuda:0')
        temp[:, :3] = to_torch(target_bodies['base_link']['info']['position'], device='cuda:0').tile((q_opt.shape[0], 1))
        temp[:, -1] = 1.        
        retarget_global_joint_pos, retarget_global_joint_quat, retarget_local_joint_pos, retarget_local_joint_quat = forward_kinematics(self.target, torch.cat((temp, q_opt), dim=-1), self.JOINT_MAPPING)
        ### FROM HEREON, CONSTRUCT THE RETARGET TRAJECTORY!!!###
        q_opt_prev, q_vel_opt = torch.zeros_like(q_opt), torch.zeros_like(q_opt)

        q_opt_prev[1:] = q_opt[:-1]
        q_vel_opt[1:] = (q_opt - q_opt_prev)[1:]/dt
        L_foot_pos_opt = retarget_local_joint_pos[FOOT_NAME[0]]
        R_foot_pos_opt = retarget_local_joint_pos[FOOT_NAME[1]]
        assert q_vel_opt.shape[0] == L_foot_pos_opt.shape[0]
        # time = torch.tensor([i for i in range(q_vel_opt.shape[0])], dtype=torch.float32, device='cuda:0').view(-1,1)
        # retarget_reference = torch.cat((retarget_global_joint_quat['virtual_joint'], q_opt, q_vel_opt, L_foot_pos_opt, R_foot_pos_opt), dim=-1)
        retarget_reference = torch.cat((self.root_ratio*reference[:, 0:1],retarget_global_joint_quat['virtual_joint'], self.root_ratio*source_local_base_lin_vel, source_local_base_ang_vel, q_opt, q_vel_opt, L_foot_pos_opt, R_foot_pos_opt), dim=-1)

        target_reference = torch.zeros((q_opt.shape[0], 3*13), requires_grad=False, device='cuda:0')
        for joint_name, _ in self.JOINT_MAPPING.items():
            target_reference[:, 3*(self.JOINT_MAPPING[joint_name]-6):3*(self.JOINT_MAPPING[joint_name]-5)] = retarget_global_joint_pos[joint_name]#target_global_joint_pos[joint_name] # retarget_global_joint_pos[joint_name]
        target_reference[:, 0:3] = retarget_global_joint_pos['virtual_joint'][:, 0:3]#target_global_joint_pos['virtual_joint'][:, 0:3] # retarget_global_joint_pos['virtual_joint'][:, 0:3]

        # # Check for foot quats
        # L_ankle_roll_joint = FOOT_NAME[0]
        # R_ankle_roll_joint = FOOT_NAME[1]
        # L_AnkleRoll_angle = q_opt[:, self.JOINT_MAPPING[L_ankle_roll_joint]-7]
        # R_AnkleRoll_angle = q_opt[:, self.JOINT_MAPPING[R_ankle_roll_joint]-7]
        # L_roll_quat = quat_from_angle_axis(L_AnkleRoll_angle ,to_torch(target_joints[L_ankle_roll_joint]['axis']))
        # R_roll_quat = quat_from_angle_axis(R_AnkleRoll_angle ,to_torch(target_joints[R_ankle_roll_joint]['axis']))

        # L_foot_euler = get_euler_xyz(retarget_global_joint_quat[L_ankle_roll_joint])
        # R_foot_euler = get_euler_xyz(retarget_global_joint_quat[R_ankle_roll_joint])
        # L_foot_euler = (normalize_angle(L_foot_euler[0]), normalize_angle(L_foot_euler[1]), normalize_angle(L_foot_euler[2]))
        # R_foot_euler = (normalize_angle(R_foot_euler[0]), normalize_angle(R_foot_euler[1]), normalize_angle(R_foot_euler[2]))
        # print("retarget L foot euler x : ", L_foot_euler[0].abs().mean())
        # print("retarget L foot euler y : ", L_foot_euler[1].abs().mean())
        # print("retarget L foot euler z : ", L_foot_euler[2].abs().mean())
        # print("retarget R foot euler x : ", R_foot_euler[0].abs().mean())
        # print("retarget R foot euler y : ", R_foot_euler[1].abs().mean())
        # print("retarget R foot euler z : ", R_foot_euler[2].abs().mean())


        # L_ankle_roll_joint = FOOT_NAME[0]
        # R_ankle_roll_joint = FOOT_NAME[1]
        # L_AnkleRoll_angle = source_qpos[:, self.JOINT_MAPPING[L_ankle_roll_joint]-7]
        # R_AnkleRoll_angle = source_qpos[:, self.JOINT_MAPPING[R_ankle_roll_joint]-7]
        # L_roll_quat = quat_from_angle_axis(L_AnkleRoll_angle ,to_torch(source_joints[L_ankle_roll_joint]['axis']))
        # R_roll_quat = quat_from_angle_axis(R_AnkleRoll_angle ,to_torch(source_joints[R_ankle_roll_joint]['axis']))

        # L_foot_euler = get_euler_xyz(source_global_joint_quat[L_ankle_roll_joint])
        # R_foot_euler = get_euler_xyz(source_global_joint_quat[R_ankle_roll_joint])
        # L_foot_euler = (normalize_angle(L_foot_euler[0]), normalize_angle(L_foot_euler[1]), normalize_angle(L_foot_euler[2]))
        # R_foot_euler = (normalize_angle(R_foot_euler[0]), normalize_angle(R_foot_euler[1]), normalize_angle(R_foot_euler[2]))
        # print("source L foot euler x : ", L_foot_euler[0].abs().mean())
        # print("source L foot euler y : ", L_foot_euler[1].abs().mean())
        # print("source L foot euler z : ", L_foot_euler[2].abs().mean())
        # print("source R foot euler x : ", R_foot_euler[0].abs().mean())
        # print("source R foot euler y : ", R_foot_euler[1].abs().mean())
        # print("source R foot euler z : ", R_foot_euler[2].abs().mean())
        
        # return target_reference.detach()
        return retarget_reference.detach(), target_reference.detach()
    
    def retarget(self, source_motion_data, play=False, iterations=2000):
        source_reference, source_info = process_data(source_motion_data)
        return self.retarget_tensor(source_reference, iterations, play)

# ###MAIN FUNCTION###

# # parser = AsfParser()
# # parser.parse(ASF_FILE)
# # parser.save_mujoco_xml(XML_FILE)

# # Create reference trajectory using AMC/ASF files
# model_processor = MotionRetarget(XML_FILE, TARGET_XML_FILE)
# q_euler, q_quat = amc_to_numpy(ASF_FILE, AMC_FILE)
# q_euler = torch.tensor(q_euler, device='cuda:0', dtype=torch.float)
# q_quat = torch.tensor(q_quat, device='cuda:0', dtype=torch.float)
# reference_length = q_euler.shape[0]
# assert q_euler.shape[1] == 18, f"q_euler shape is {q_euler.shape}"
# assert q_quat.shape[1] == 19, f"q_quat shape is {q_quat.shape}"
# global_joint_pos, global_joint_quat, local_joint_pos, local_joint_quat = forward_kinematics(model_processor.source, q_quat, model_processor.JOINT_MAPPING)
# L_foot_pos = local_joint_pos[FOOT_NAME[0]]
# R_foot_pos = local_joint_pos[FOOT_NAME[1]]
# L_foot_rpy = quaternion_to_euler(local_joint_quat[FOOT_NAME[0]])
# R_foot_rpy = quaternion_to_euler(local_joint_quat[FOOT_NAME[1]])
# q_pose = q_euler[:, 6:]
# q_prev, q_vel = torch.zeros_like(q_pose), torch.zeros_like(q_pose)
# q_prev[1:, :] = q_pose[:-1, :]
# q_vel[1:, :] = (q_pose[1:, :] - q_prev[1:, :]) / dt
# q_vel[0, :] = 0.
# base_pos = q_euler[:, :3]
# base_euler = q_euler[:, 3:6]
# base_linvel, base_angvel = torch.zeros_like(base_pos), torch.zeros_like(base_euler)
# base_linvel[1:, :], base_angvel[1:, :] = (base_pos[1:, :] - base_pos[:-1, :])/dt, euler_to_angular_velocity(base_euler, dt)
# base_linvel[0,:], base_angvel[0,:] = 0.,0. 
# time_stamp = torch.tensor([i*dt for i in range(reference_length)], dtype=torch.float, device='cuda:0')
# reference_tensor = torch.cat((time_stamp.unsqueeze(1), base_pos, base_euler, base_linvel, base_angvel, q_pose, q_vel, L_foot_pos, L_foot_rpy,  R_foot_pos, R_foot_rpy), dim=-1)
# np.savetxt(TXT_FILE, reference_tensor.detach().cpu().numpy(), fmt='%f', delimiter='    ')

# # Read the input file
# with open(TXT_FILE, 'r') as file:
#     lines = file.readlines()

# # Process each line to format it as a list
# formatted_lines = []
# for line in lines:
#     # Remove any leading/trailing whitespace
#     line = line.strip()
#     # Skip empty lines
#     if not line:
#         continue
#     # Split the line by spaces to get individual numbers
#     numbers = line.split()
#     # Join the numbers with commas
#     numbers_with_commas = ', '.join(numbers)
#     # Add brackets if not already present
#     if not line.startswith('['):
#         numbers_with_commas = '[' + numbers_with_commas
#     if not line.endswith(']'):
#         numbers_with_commas = numbers_with_commas + '],'
#     # Append the formatted line to the list
#     formatted_lines.append(numbers_with_commas)

# # Write the formatted lines to the output file
# with open(JSON_FILE, 'w') as file:
#     file.write('{"MotionWeight": 1.0,\n"Frames":[\n')
#     if formatted_lines:
#         formatted_lines[-1] = formatted_lines[-1].rstrip(',') 
#     for line in formatted_lines:
#         file.write(line + '\n')
#     file.write(']}')

# # Retarget reference trajectory according to TARGET_XML_FILE
# for i, motion_file in enumerate(glob.glob(JSON_FILE)):
#     with open(motion_file, "r") as f:
#         motion_json = json.load(f)
#         motion_data = np.array(motion_json["Frames"]) # motion data in json form.

# retarget = MotionRetarget(source_model_path=XML_FILE, target_model_path=TARGET_XML_FILE)
# retarget_reference, target_reference = retarget.retarget(motion_data)
# write_tensor_to_txt(retarget_reference, RETARGET_FILE)
# print("Retarget reference written")