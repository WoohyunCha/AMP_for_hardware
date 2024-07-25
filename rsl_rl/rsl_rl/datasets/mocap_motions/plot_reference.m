% Define the file path
filename = 'tocabi_data_scaled.txt';

% Open the file for reading
fileID = fopen(filename, 'r');

% Read data from the file
data = textscan(fileID, repmat('%f', 1, 91), 'Delimiter', ' ');
% Close the file
fclose(fileID);

% Extract columns from the cell array
time = data{1};        % Time stamps
indices = time > 5.6 & time < 7.4005; % Logical indices for the required time range
time = time(indices);
value2 = data{2}(indices);      % Second values, filtered
value3 = data{3}(indices);      % Third values, filtered
value4 = data{4}(indices);      % Fourth values, filtered

% Create figure window
figure(1);

% Plot second value against time
subplot(3, 1, 1);
plot(time, value2);
title('X position vs. Time');
xlabel('Time');
ylabel('X position');

% Plot third value against time
subplot(3, 1, 2);
plot(time, value3);
title('Y position vs. Time');
xlabel('Time');
ylabel('Y position');

% Plot fourth value against time
subplot(3, 1, 3);
plot(time, value4);
title('Z position vs. Time');
xlabel('Time');
ylabel('Z position');

% Enhance layout
sgtitle('Plots of Positions Against Time');
