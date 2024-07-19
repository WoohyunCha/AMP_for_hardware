import json

# Define the input and output file paths
file_name = 'processed_data_tocabi_walk'
input_file_path = file_name+'.txt'
output_file_path = file_name+'.json'

# Read the input file
with open(input_file_path, 'r') as file:
    lines = file.readlines()

# Process each line to format it as a list
formatted_lines = []
for line in lines:
    # Remove any leading/trailing whitespace
    line = line.strip()
    # Skip empty lines
    if not line:
        continue
    # Split the line by spaces to get individual numbers
    numbers = line.split()
    # Join the numbers with commas
    numbers_with_commas = ', '.join(numbers)
    # Add brackets if not already present
    if not line.startswith('['):
        numbers_with_commas = '[' + numbers_with_commas
    if not line.endswith(']'):
        numbers_with_commas = numbers_with_commas + '],'
    # Append the formatted line to the list
    formatted_lines.append(numbers_with_commas)

# Write the formatted lines to the output file
with open(output_file_path, 'w') as file:
    file.write('{"Frames":[\n')
    for line in formatted_lines:
        file.write(line + '\n')
    file.write(']}\n')

print("Conversion completed. The formatted data is saved in output.txt.")
