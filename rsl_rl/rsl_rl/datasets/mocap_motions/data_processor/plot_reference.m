% Define the file path
filename = '../AMP_trajectories/processed_data_2.txt';

% Open the file for reading
fileID = fopen(filename, 'r');

% Read data from the file
data = textscan(fileID, repmat('%f', 1, 49), 'Delimiter', ' ');
% Close the file
fclose(fileID);

% Convert cell array to matrix
data = cell2mat(data);

% Define time (data collected at 2000 Hz)
time = (0:size(data,1)-1)'/2000;

% Extract data categories
base_height = data(:,1);
base_quaternion = data(:,2:5);
base_linear_velocity = data(:,6:8);
base_angular_velocity = data(:,9:11);
joint_positions = data(:,12:23);
joint_velocities = data(:,24:35);
foot_positions = data(:,36:41);
foot_quaternions = data(:,42:49);

% Plotting
figure;

% Base height
subplot(4,2,1);
plot(time, base_height);
title('Base Height');
xlabel('Time (s)');
ylabel('Height (m)');

% Base linear velocity
subplot(4,2,2);
plot(time, base_linear_velocity);
legend('X', 'Y', 'Z');
title('Base Linear Velocity');
xlabel('Time (s)');
ylabel('Velocity (m/s)');

% Base angular velocity
subplot(4,2,3);
plot(time, base_angular_velocity);
legend('Roll', 'Pitch', 'Yaw');
title('Base Angular Velocity');
xlabel('Time (s)');
ylabel('Angular velocity (rad/s)');

% Joint positions
subplot(4,2,4);
plot(time, joint_positions);
title('Joint Positions');
xlabel('Time (s)');
ylabel('Position (rad)');

% Joint velocities
subplot(4,2,5);
plot(time, joint_velocities);
title('Joint Velocities');
xlabel('Time (s)');
ylabel('Velocity (rad/s)');

% Foot positions
subplot(4,2,6);
plot(time, foot_positions(:,1:3));
hold on;
plot(time, foot_positions(:,4:6));
legend('Left X', 'Left Y', 'Left Z', 'Right X', 'Right Y', 'Right Z');
title('Foot Positions');
xlabel('Time (s)');
ylabel('Position (m)');

% Foot quaternions
subplot(4,2,7);
plot(time, foot_quaternions(:,1:4));
hold on;
plot(time, foot_quaternions(:,5:8));
legend('Left W', 'Left X', 'Left Y', 'Left Z', 'Right W', 'Right X', 'Right Y', 'Right Z');
title('Foot Quaternions');
xlabel('Time (s)');
ylabel('Quaternion');

% Adjust layout to prevent label overlap
sgtitle('Sensor Data Overview');
tight_layout();
