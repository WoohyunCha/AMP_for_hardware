import xml.etree.ElementTree as ET
import os
import numpy as np


def _create_robot_asset(input_file, output_file):
    source_asset_path = input_file
    source_asset_root = os.path.dirname(source_asset_path)
    source_asset_file = os.path.basename(source_asset_path)
    target_asset_file = os.path.basename(output_file)
    target_asset_path = output_file
    link_length_randomize_range = 0.5
    # set randomized link lengths
    hip_randomize_scale = 1 + (2*np.random.random() - 1) * link_length_randomize_range
    thigh_randomize_scale = 1 + (2*np.random.random() - 1) * link_length_randomize_range
    shin_randomize_scale = 1 + (2*np.random.random() - 1) * link_length_randomize_range
    ankle_randomize_scale = 1 + (2*np.random.random() - 1) * link_length_randomize_range
    tree = ET.parse(source_asset_path)
    root = tree.getroot()
    init_height = 0.
    default_angle = 0. * np.pi/180.

    # Iterate through all 'body' elements
    for body in root.findall('.//body'):
        body_name = body.get('name')

        # Check if 'HipRoll_Link' is in the body's name
        if 'Hip' in body_name:
            # Fix the last component of 'pos' of the body
            if 'pos' in body.attrib:
                pos_values = body.get('pos').split()
                pos_values[-1] = str(float(pos_values[-1]) * hip_randomize_scale)  # Replace the last component with A
                init_height += float(pos_values[-1])
                body.set('pos', ' '.join(pos_values))

            # Fix the last component of 'pos' of the 'inertial' element under this body
            inertial = body.find('inertial')
            if inertial is not None and 'pos' in inertial.attrib:
                inertial_pos_values = inertial.get('pos').split()
                inertial_pos_values[-1] = str(float(inertial_pos_values[-1]) * hip_randomize_scale)  # Replace the last component with B
                inertial.set('pos', ' '.join(inertial_pos_values))
            for geom in body.findall("./geom[@class='cls']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * hip_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values)) 
            for geom in body.findall("./geom[@class='cls']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * hip_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values)) 
            for geom in body.findall("./geom[@class='viz']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * hip_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values))
            for geom in body.findall("./geom[@class='viz']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * hip_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values))
        elif 'Thigh' in body_name:
            if 'pos' in body.attrib:
                pos_values = body.get('pos').split()
                pos_values[-1] = str(float(pos_values[-1]) * hip_randomize_scale)  # Replace the last component with A
                body.set('pos', ' '.join(pos_values))
                init_height += float(pos_values[-1])# * np.cos(default_angle)

            # Fix the last component of 'pos' of the 'inertial' element under this body
            inertial = body.find('inertial')
            if inertial is not None and 'pos' in inertial.attrib:
                inertial_pos_values = inertial.get('pos').split()
                inertial_pos_values[-1] = str(float(inertial_pos_values[-1]) * thigh_randomize_scale)  # Replace the last component with B
                inertial.set('pos', ' '.join(inertial_pos_values))
            for geom in body.findall("./geom[@class='cls']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * thigh_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values)) 
            for geom in body.findall("./geom[@class='cls']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * thigh_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values)) 
            for geom in body.findall("./geom[@class='viz']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * thigh_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values))
            for geom in body.findall("./geom[@class='viz']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * thigh_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values))
        elif 'Knee' in body_name:
            if 'pos' in body.attrib:
                pos_values = body.get('pos').split()
                pos_values[-1] = str(float(pos_values[-1]) * thigh_randomize_scale)  # Replace the last component with A
                body.set('pos', ' '.join(pos_values))
                init_height += float(pos_values[-1])  #* np.cos(default_angle)

            # Fix the last component of 'pos' of the 'inertial' element under this body
            inertial = body.find('inertial')
            if inertial is not None and 'pos' in inertial.attrib:
                inertial_pos_values = inertial.get('pos').split()
                inertial_pos_values[-1] = str(float(inertial_pos_values[-1]) * shin_randomize_scale)  # Replace the last component with B
                inertial.set('pos', ' '.join(inertial_pos_values))
            for geom in body.findall("./geom[@class='cls']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * shin_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values)) 
            for geom in body.findall("./geom[@class='cls']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * shin_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values)) 
            for geom in body.findall("./geom[@class='viz']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * shin_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values))
            for geom in body.findall("./geom[@class='viz']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * shin_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values))
        elif 'AnkleCenter' in body_name:
            if 'pos' in body.attrib:
                pos_values = body.get('pos').split()
                pos_values[-1] = str(float(pos_values[-1]) * shin_randomize_scale)  # Replace the last component with A
                body.set('pos', ' '.join(pos_values))
                init_height += float(pos_values[-1])

            # Fix the last component of 'pos' of the 'inertial' element under this body
            inertial = body.find('inertial')
            if inertial is not None and 'pos' in inertial.attrib:
                inertial_pos_values = inertial.get('pos').split()
                inertial_pos_values[-1] = str(float(inertial_pos_values[-1]) * ankle_randomize_scale)  # Replace the last component with B
                inertial.set('pos', ' '.join(inertial_pos_values))
            for geom in body.findall("./geom[@class='cls']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values)) 
            for geom in body.findall("./geom[@class='cls']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values)) 
            for geom in body.findall("./geom[@class='viz']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values))
            for geom in body.findall("./geom[@class='viz']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values))

        # Check if 'Foot_Link' is in the body's name
        if 'AnkleRoll' in body_name:
            # Fix the last component of 'pos' of the 'geom' element with class 'cls' under this body
            for inertial in body.findall(".//inertial"):
                if 'pos' in inertial.attrib:
                    inertial_pos_values = inertial.get('pos').split()
                    inertial_pos_values[-1] = str(float(inertial_pos_values[-1]) * ankle_randomize_scale)  # Replace the last component with B
                    inertial.set('pos', ' '.join(inertial_pos_values))
            for geom in body.findall(".//geom[@class='cls']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values)) 
            for geom in body.findall(".//geom[@class='cls']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values)) 
            for geom in body.findall(".//geom[@class='viz']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    geom_pos_values[-1] = str(float(geom_pos_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('pos', ' '.join(geom_pos_values))
            for geom in body.findall(".//geom[@class='viz']"):
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    geom_sz_values[-1] = str(float(geom_sz_values[-1]) * ankle_randomize_scale)  # Replace the last component with C
                    geom.set('size', ' '.join(geom_sz_values))
        
        if 'Foot_Link' in body_name:
            for geom in body.findall("./geom[@class='cls']"):
                if 'pos' in geom.attrib:
                    geom_pos_values = geom.get('pos').split()
                    init_height += float(geom_pos_values[-1])
                if 'size' in geom.attrib:
                    geom_sz_values = geom.get('size').split()
                    init_height+= float(geom_sz_values[-1])
    # Write the modified XML to the output file

    for body in root.findall('.//body[@name="base_link"]'):
    # Get the current position attribute
        current_pos_values = body.get('pos').split()
        current_pos_values[-1] = str(-0.5*init_height)  # Replace the last component with C
        body.set('pos', ' '.join(current_pos_values))
        # Print the new position (optional)
        print(f"New position: {current_pos_values}")
    tree.write(target_asset_path, encoding='utf-8', xml_declaration=True)

# Usage example
input_xml = "/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi_nomesh.xml"
output_xml = "/home/cha/isaac_ws/AMP_for_hardware/resources/robots/tocabi/xml/dyros_tocabi_random.xml"
_create_robot_asset(input_xml, output_xml)
