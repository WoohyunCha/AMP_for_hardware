from lxml import etree
import numpy as np

MJCF_file = '/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi.xml'

class MotionRetarget():
    def __init__(self, source_model_path: str, source_reference: np.ndarray):
        self.source = source_model_path
        self.reference = source_reference
        self.source_bodies, self.source_joints, self.source_edges = self.process_model(self.source)
        
        self.print_model(self.source)

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
                    assert i == 0, f"Multiple parent link is prohibited, parent link {i} detected."
                    joint_name = joint.get('name')
                    joint_pos = np.array([float(num) for num in joint.get('pos', '0 0 0').split()], dtype=float)
                    joint_axis = np.array([float(num) for num in joint.get('axis', '0 0 1').split()], dtype=float)
                    joint_type = joint.get('type', 'revolute')
                    joint_range = np.array([float(num) for num in joint.get('range', '-3.14 3.14').split()], dtype=float)
                    joint_info = {
                        'parent' : parent_name,
                        'child' : body_name,
                        'position' : joint_pos,
                        'axis' : joint_axis,
                        'type' : joint_type,
                        'range' : joint_range
                    }
                    joints[joint_name] = joint_info
            else: # fixed link
                joint_name = None             
            bodies[body_name] = {'parent': parent_name, 'joint': joint_name, 'info' : body_info}
            if (body_name, parent_name) not in edges:
                edges[(body_name, parent_name)] = {}
            if (parent_name, body_name) not in edges:
                edges[(parent_name, body_name)] = {}

        # process link lengths
        for body_name, val in bodies.items():
            pos = val['info']['position']
            parent_name = val['parent']
            edges[(body_name, parent_name)]['length'] = np.linalg.norm(pos, ord=2, axis=-1)
            edges[(parent_name, body_name)]['length'] = np.linalg.norm(pos, ord=2, axis=-1)
       
        
        return bodies, joints, edges

    def print_model(self, model_path):
        bodies, joints, edges = self.process_model(model_path=model_path)
        print("PRINTING MODEL")
        print("---------------INFO----------------")
        for body_name, val in bodies.items():
            print("Link name : ", body_name)
            print(val)
            print("Joint name : ", val['joint'])
            print(joints[val['joint']]) if val['joint'] is not None else print("None")
            print("----------------------")
        print("-------------Link length-----------")
        for tuple, val in edges.items():
            print(f"{tuple} : {val}")

    def retarget(self, target_model_path: str, target_referece: np.ndarray) -> np.ndarray: # outputs retargetted reference
        target_bodies, target_joints, target_edges = self.process_model(target_model_path)
        length_ratios = {}
        for key, val in target_edges:
            length_ratios[key] = val/self.source_edges[key]
        retarget_reference = np.zeros_like(target_referece)
        for frame_idx in range(target_referece.shape[0]):
            retarget_reference[frame_idx]
        #TODO
        # 1. FK to get global positions of each joint (we have global pos/ori of base link)
        # 2. Compute relative position vectors between joints, and scale them by length ratios
        # 3. Starting from base link, sum up the relative position vectors and transform back to base frame coordinates
        # 4. Solve numerical IK from joint positions.

##########HELPER###########
def quat_from_angle_axis(angle, axis: np.ndarray):
    theta = (angle / 2).expand_dims(-1)
    xyz = normalize(axis) * np.sin(theta)
    w = np.cos(theta)
    return quat_unit(np.concatenate((xyz, w), axis=-1))

def normalize(x: np.ndarray, eps: float = 1e-9):
    return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True).clip(eps, None)

def quat_unit(a):
    return normalize(a)

def main():
    mjcf_file_path = MJCF_file
    MotionRetarget(mjcf_file_path, None)

if __name__ == '__main__':
    main()
