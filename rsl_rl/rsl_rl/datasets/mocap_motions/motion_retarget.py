import xml.etree.ElementTree as ET
import numpy as np

def parse_mjcf(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    joints = []
    bodies = {}

    # Collect body and joint data
    for body in root.findall('.//body'):
        body_name = body.get('name')
        pos = np.array(body.get('pos', '0 0 0').split(), dtype=float) # default value is 0, 0, 0
        quat = np.array(body.get('quat', '0 0 0 1').split(), dtype=float)
        bodies[body_name] = {'pos': pos, 'quat': quat, 'children': []}

        for joint in body.findall('joint'):
            joint_name = joint.get('name')
            joint_pos = np.array(joint.get('pos', '0 0 0').split(), dtype=float)
            joint_axis = np.array(joint.get('axis', '0 0 1').split(), dtype=float)
            joint_range = np.array(joint.get('range', '-3.14, 3.14').split(), dtype=float)
            joints.append((body_name, joint_name, joint_pos))

            # Link to parent body
            if body_name in bodies:
                bodies[body_name]['children'].append((joint_name, joint_pos))

    return bodies, joints

def compute_kinematics(bodies):
    # Compute forward kinematics
    for body_name, body_info in bodies.items():
        parent_pos = body_info['pos']
        for child_name, child_pos in body_info['children']:
            # Child absolute position in base frame coordinates
            child_absolute_pos = parent_pos + child_pos
            print(f"{child_name} position relative to base: {child_absolute_pos}")

            # Link vector calculation (if it has a parent, and is not the root)
            if 'parent' in body_info:
                link_vector = child_absolute_pos - bodies[body_info['parent']]['pos']
                print(f"Link vector from {body_info['parent']} to {child_name}: {link_vector}")

def main():
    mjcf_file_path = 'path_to_your_mjcf_file.xml'
    bodies, joints = parse_mjcf(mjcf_file_path)
    compute_kinematics(bodies)

if __name__ == '__main__':
    main()
