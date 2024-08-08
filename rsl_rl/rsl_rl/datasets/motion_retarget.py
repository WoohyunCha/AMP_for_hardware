from isaacgym import gymtorch, gymapi, gymutil

from lxml import etree
import numpy as np
from rsl_rl.utils import utils
from rsl_rl.datasets import pose3d
from rsl_rl.datasets import motion_util

from isaacgym.torch_utils import *
from rsl_rl.datasets.motion_loader import AMPLoader
import glob
import json
import torch.optim as optim

MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi.xml'
source_MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi.xml'
# target_MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi.xml'
# source_MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/cmu.xml'
target_MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/data/raw/CMU_open/cmu.xml'
JSON_file = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/tocabi_data_scaled_1_0x.json'
RETARGET_file = '/home/cha/isaac_ws/AMP_for_hardware/rsl_rl/rsl_rl/datasets/mocap_motions/retarget_motions/retarget_reference_data.txt'

FRAME_TIME = 0
MODEL_DOF = 33

POS_SIZE = 3
JOINT_POS_SIZE = 12
LINEAR_VEL_SIZE = 3
ANGULAR_VEL_SIZE = 3
JOINT_VEL_SIZE = 12
ROOT_POS_SIZE = 1 # base height
ROOT_ROT_SIZE = 4 

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

HZ = 2000
# REFERENCE_START_INDEX = [int(5.6*HZ), int(4.4*HZ)]
# REFERENCE_END_INDEX = [int(7.4005*HZ), int(5.6005*HZ)]
REFERENCE_START_INDEX = int(2.0*HZ)
REFERENCE_END_INDEX = int(7.4005*HZ)

BASE_PELVIS_OFFSET = (0., 0., 0.)

FOOT_JOINTS = ['L_AnklePitch_Joint', 'R_AnklePitch_Joint', 'L_AnkleRoll_Joint', 'R_AnkleRoll_Joint'] # Used to calculate foot quaternion
VIRTUAL_JOINT_DIM = 7

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
        # print("SOURCE MODEL")
        # self.print_model(self.source)
        # print("TARGET MODEL")
        # self.print_model(self.target)

    def process_model(self, model_path: str):
        self.source = model_path
        tree = etree.parse(self.source)
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

    def retarget(self, reference: torch.Tensor) -> torch.Tensor: # outputs retargetted reference
        length_ratios = {}
        reference_length = reference.shape[0]
        observation_dim = reference.shape[-1]
        for key, val in self.target['edges'].items():
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
        source_global_root_pos = torch.cat((torch.zeros_like(reference[:, start:end]), torch.zeros_like(reference[:, start:end]), reference[:, start:end]), dim=-1)
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
            # if body_name == "base_link":
            #     source_global_joint_pos[joint_name] = source_global_root_pos + quat_rotate(source_global_root_quat, to_torch(source_joints[joint_name]['position']).tile((reference_length,1)) ) # g_P_j/o = g_P_b/o + g_R_b*b_P_j/b
            #     source_global_joint_quat[joint_name] = source_global_root_quat
            #     base_global_pos = source_global_joint_pos[joint_name] # base position is same as virtual joint in mjcf
            #     base_global_quat:torch.Tensor = source_global_joint_quat[joint_name] # base quat is same as virtual joint in mjcf
            #     source_global_Lfoot_quat = quat_mul(base_global_quat, source_local_Lfoot_quat)
            #     source_global_Lfoot_pos = base_global_pos + quat_rotate(source_global_Lfoot_quat, source_local_Lfoot_pos)
            #     source_global_Rfoot_quat = quat_mul(base_global_quat, source_local_Rfoot_quat)
            #     source_global_Rfoot_pos = base_global_pos + quat_rotate(source_global_Rfoot_quat, source_local_Rfoot_pos)
            # elif joint_name in self.JOINT_MAPPING:
            #     # print("---------------------------")
            #     # print("Joint : ", joint_name)
            #     parent_joint_name = source_bodies[parent_name]['joint']
            #     # print("Parent joint : ", parent_joint_name)
            #     source_global_joint_pos_parent = source_global_joint_pos[parent_joint_name] # parent's joint's position in global coordinates
            #     # print("Parent joint global pos : ", source_global_joint_pos_parent[1000])
            #     source_global_joint_quat_parent = source_global_joint_quat[parent_joint_name] # parent's joint's quaternion in global coordinates (inlcuding rotation due to joint actuation)
            #     # print("Parent joint global quat : ", source_global_joint_quat_parent[1000])
            #     source_joint_pose = source_qpos[:, self.JOINT_MAPPING[joint_name]] # joint angle that connects body and its parent
            #     # print("Mapping index of joint : ", self.JOINT_MAPPING[joint_name])
            #     source_global_joint_quat[joint_name] = quat_mul(source_global_joint_quat_parent, quat_from_angle_axis(source_joint_pose, to_torch(source_joints[joint_name]['axis']).tile((reference_length, 1))))
            #     # print("Joint global quat : ", source_global_joint_quat[joint_name][1000])
            #     source_global_joint_pos[joint_name] = source_global_joint_pos_parent + quat_rotate(source_global_joint_quat_parent, to_torch(body_info['position']).tile((reference_length,1))) # source_global_joint_quat_parent * parent_body_info['position']

                # 2. Compute relative position vectors between joints and scale them by length ratios
                # assert to_torch(length_ratios[joint_name]).tile((source_global_joint_pos_parent.shape[0], 1)).shape == (reference_length, 1), f"length ratios tile is wrong. Became shape {to_torch(length_ratios[joint_name]).tile((source_global_joint_pos_parent.shape[0], 1)).shape}"

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
        weight_joint_pos, weight_foot_quat, weight_joint_pose = 1., 1., .1
        weight_norm = (weight_joint_pos+weight_foot_quat+weight_joint_pose)
        weight_joint_pos /= weight_norm
        weight_foot_quat /= weight_norm
        weight_joint_pose/= weight_norm

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

            joint_pose_cost = torch.nn.MSELoss()(x, reference[:, JOINT_POSE_START_IDX:JOINT_POSE_END_IDX])
            joint_pos_cost = torch.nn.MSELoss()(x_local_joint_pos_tensor, joint_pos_local_retarget)

            foot_quat_local_target_inverse = reference[:, ROOT_ROT_START_IDX:ROOT_ROT_END_IDX] # We want the foot to be flat to the ground. b_R_foot = b_R_g = g_R_b^-1
            L_ankle_pitch_joint = FOOT_JOINTS[0]
            R_ankle_pitch_joint = FOOT_JOINTS[1]
            L_ankle_roll_joint = FOOT_JOINTS[2]
            R_ankle_roll_joint = FOOT_JOINTS[3]
            x_Lfoot_joint_quat = x_global_joint_quat[L_ankle_pitch_joint] # Ankle pitch joint quat
            x_Rfoot_joint_quat = x_global_joint_quat[R_ankle_pitch_joint] # Ankle pitch joint quat
            L_ankle_pitch_quat = quat_from_angle_axis(x[:, self.JOINT_MAPPING[L_ankle_pitch_joint]-7] ,to_torch(target_joints[L_ankle_pitch_joint]['axis']))
            R_ankle_pitch_quat = quat_from_angle_axis(x[:, self.JOINT_MAPPING[R_ankle_pitch_joint]-7] ,to_torch(target_joints[R_ankle_pitch_joint]['axis']))
            L_ankle_roll_quat = quat_from_angle_axis(x[:, self.JOINT_MAPPING[L_ankle_roll_joint]-7] ,to_torch(target_joints[L_ankle_roll_joint]['axis']))
            R_ankle_roll_quat = quat_from_angle_axis(x[:, self.JOINT_MAPPING[R_ankle_roll_joint]-7] ,to_torch(target_joints[R_ankle_roll_joint]['axis']))

            x_Lfoot_quat = x_global_joint_quat[FOOT_JOINTS[1]]
            x_Rfoot_quat = x_global_joint_quat[FOOT_JOINTS[3]]
            # print("global frame in base coordinates : ", quat_conjugate(foot_quat_local_target_inverse))
            # print("Lfoot quat : ", x_Lfoot_quat)
            # print("Lfoot quat distance : ", distance_between_quats(quat_conjugate(foot_quat_local_target_inverse), x_Lfoot_quat, neglect_axis=2))
            # print("Rfoot quat : ", x_Rfoot_quat)
            # print("Rfoot quat distance : ", distance_between_quats(quat_conjugate(foot_quat_local_target_inverse), x_Rfoot_quat, neglect_axis=2))
            zero_quat = torch.tensor([0, 0, 0, 1], dtype=torch.float, device='cuda:0', requires_grad=False).tile(reference_length,1)
            foot_quat_cost = (distance_between_quats(zero_quat, x_Lfoot_quat, neglect_axis=2).pow(2) \
                              + distance_between_quats(zero_quat, x_Rfoot_quat, neglect_axis=2).pow(2)).mean()
            # print("joint pos cost mean : ", joint_pos_cost.mean())
            # print("joint pose cost mean : ", joint_pose_cost.mean())
            # print("foot quat cost mean : ", foot_quat_cost.mean())
            total_cost = weight_joint_pos*joint_pos_cost + weight_joint_pose*joint_pose_cost + weight_foot_quat*foot_quat_cost
            return torch.sum(total_cost)
        
        n_iterations = 3000
        q_opt = torch.zeros((reference_length, 12), dtype=torch.float32, requires_grad=True, device='cuda:0')
        
        q_opt = reference[:, JOINT_POSE_START_IDX:JOINT_POSE_END_IDX].clone().detach().requires_grad_(True)
        optimizer = optim.Adam([q_opt], lr=1e-4)
        print("SOLVING IK....")
        for i in range(n_iterations):
            optimizer.zero_grad()
            cost = cost_function(q_opt)
            cost.backward()
            optimizer.step()
            if i%100 == 0:
                print(f"Iteration {i}, Cost: {cost.item()}")

            # Optionally, you can add some stopping criterion based on the change in cost or other conditions
            if cost.item() < 1e-4:
                print("Stopping criterion met")
                break
        assert ~torch.any(~torch.isfinite(q_opt)), "Cannot solve IK!"
        print("IK SOLVED!")

        temp = torch.zeros((q_opt.shape[0], 7), requires_grad=False, device='cuda:0')
        temp[:, :3] = to_torch(target_bodies['base_link']['info']['position'], device='cuda:0').tile((q_opt.shape[0], 1))
        temp[:, -1] = 1.        
        retarget_global_joint_pos, retarget_global_joint_quat, retarget_local_joint_pos, retarget_local_joint_quat = forward_kinematics(self.target, torch.cat((temp, q_opt), dim=-1), self.JOINT_MAPPING)
        target_reference = torch.zeros((q_opt.shape[0], 3*13), requires_grad=False, device='cuda:0')
        for joint_name, _ in self.JOINT_MAPPING.items():
            target_reference[:, 3*(self.JOINT_MAPPING[joint_name]-6):3*(self.JOINT_MAPPING[joint_name]-5)] = retarget_global_joint_pos[joint_name]
        target_reference[:, 0:3] = retarget_global_joint_pos['virtual_joint'][:, 0:3]

        # Check for foot quats
        L_ankle_roll_joint = FOOT_JOINTS[1]
        R_ankle_roll_joint = FOOT_JOINTS[3]
        L_AnkleRoll_angle = q_opt[:, self.JOINT_MAPPING[L_ankle_roll_joint]-7]
        R_AnkleRoll_angle = q_opt[:, self.JOINT_MAPPING[R_ankle_roll_joint]-7]
        L_roll_quat = quat_from_angle_axis(L_AnkleRoll_angle ,to_torch(target_joints[L_ankle_roll_joint]['axis']))
        R_roll_quat = quat_from_angle_axis(R_AnkleRoll_angle ,to_torch(target_joints[R_ankle_roll_joint]['axis']))

        L_foot_euler = get_euler_xyz(retarget_global_joint_quat[L_ankle_roll_joint])
        R_foot_euler = get_euler_xyz(retarget_global_joint_quat[L_ankle_roll_joint])
        L_foot_euler = (normalize_angle(L_foot_euler[0]), normalize_angle(L_foot_euler[1]), normalize_angle(L_foot_euler[2]))
        R_foot_euler = (normalize_angle(R_foot_euler[0]), normalize_angle(R_foot_euler[1]), normalize_angle(R_foot_euler[2]))
        print("retarget L foot euler : ", L_foot_euler)
        print("retarget R foot euler : ", R_foot_euler)


        L_ankle_roll_joint = FOOT_JOINTS[1]
        R_ankle_roll_joint = FOOT_JOINTS[3]
        L_AnkleRoll_angle = source_qpos[:, self.JOINT_MAPPING[L_ankle_roll_joint]-7]
        R_AnkleRoll_angle = source_qpos[:, self.JOINT_MAPPING[R_ankle_roll_joint]-7]
        L_roll_quat = quat_from_angle_axis(L_AnkleRoll_angle ,to_torch(source_joints[L_ankle_roll_joint]['axis']))
        R_roll_quat = quat_from_angle_axis(R_AnkleRoll_angle ,to_torch(source_joints[R_ankle_roll_joint]['axis']))

        L_foot_euler = get_euler_xyz(source_global_joint_quat[L_ankle_roll_joint])
        R_foot_euler = get_euler_xyz(source_global_joint_quat[R_ankle_roll_joint])
        L_foot_euler = (normalize_angle(L_foot_euler[0]), normalize_angle(L_foot_euler[1]), normalize_angle(L_foot_euler[2]))
        R_foot_euler = (normalize_angle(R_foot_euler[0]), normalize_angle(R_foot_euler[1]), normalize_angle(R_foot_euler[2]))
        print("source L foot euler : ", L_foot_euler)
        print("source R foot euler : ", R_foot_euler)

        return target_reference.detach()
        # Optimize cost_function about x

        # Cost should include setting feet orientation flat, to make the motion more feasible for robots
        # After solving for q, compute FK to get vector from lowest ankle joint to base link. Foot height + offset = base height.
            # 5. Offset base height so that foot is in contact with the ground. How will we get the offset?
        #TODO
        # 1. FK to get global positions of each joint (we have global pos/ori of base link)
        # 2. Compute relative position vectors between joints, and scale them by length ratios
        # 3. Starting from base link, sum up the relative position vectors and transform back to base frame coordinates
        # 4. Solve numerical IK from joint positions.



def main():
    amploader = AMPLoader('cuda:0', 1/250, motion_files=glob.glob(JSON_file), model_file=MJCF_file)
    for i, motion_file in enumerate(glob.glob(JSON_file)):
        with open(motion_file, "r") as f:
            motion_json = json.load(f)
            motion_data = np.array(motion_json["Frames"]) # motion data in json form.

    source_reference = amploader.process_data(motion_data, 0)
    retarget = MotionRetarget(source_model_path=source_MJCF_file, target_model_path=target_MJCF_file)
    retarget_reference = retarget.retarget(source_reference)
    write_tensor_to_txt(retarget_reference, RETARGET_file)
    print("Retarget reference written")


######HELPER#######

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


if __name__ == '__main__':
    main()