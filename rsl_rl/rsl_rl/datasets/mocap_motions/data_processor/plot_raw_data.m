% Define the file path
filename = '../data/raw/tocabi_data0.6.txt';

% Open the file for reading
fileID = fopen(filename, 'r');

% Read data from the file, assuming 92 floating-point numbers per line
data = textscan(fileID, repmat('%f', 1, 91), 'Delimiter', ' ');

% Close the file
fclose(fileID);

% Extract columns from the cell array
time = data{1}/2000;        % Time stamps
value2 = data{8};      % Second values
value3 = data{9};      % Third values
value4 = data{10};      % Fourth values

% Create figure window
figure(1);

% Plot second value against time
subplot(3, 1, 1);
plot(time, value2);
title('Second Value vs. Time');
xlabel('Time');
ylabel('Second Value');

% Plot third value against time
subplot(3, 1, 2);
plot(time, value3);
title('Third Value vs. Time');
xlabel('Time');
ylabel('Third Value');

% Plot fourth value against time
subplot(3, 1, 3);
plot(time, value4);
title('Fourth Value vs. Time');
xlabel('Time');
ylabel('Fourth Value');

% Enhance layout
sgtitle('Plots of Values Against Time 0.6');
