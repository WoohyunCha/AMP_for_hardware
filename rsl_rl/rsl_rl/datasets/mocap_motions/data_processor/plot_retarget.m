% Define the file path
filename = '../retarget_motions/retarget_reference_data.txt';

% Open the file for reading
fileID = fopen(filename, 'r');

% Prepare a figure for the animation
figure;
hold on;
grid on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Biped Robot Animation');

% Define uniform axis limits
axLimit = 1;  % Adjust this value to cover the range of your data
xlim([-axLimit axLimit]);
ylim([-axLimit axLimit]);
zlim([-1 1]);

% Set the same scaling for all axes
axis equal;
axis manual;  % Keep the axis limits fixed across frames
view(3);

% Initialize frame counter
frameCounter = 0;

% Initialize matrix to store data
dataMatrix = [];

% Animation loop
while ~feof(fileID)
    % Read one line of the file
    line = fgetl(fileID);
    
    % Increment frame counter
    frameCounter = frameCounter + 1;
    
    % Parse the line into an array of numbers
    data = str2double(strsplit(line));
    
    % Store the data in the matrix
    dataMatrix = [dataMatrix; data];  %#ok<AGROW>
    
    % Only plot every 1000th frame
    if mod(frameCounter, 1) == 0
        % Extract 3D coordinates for each joint
        nJoints = length(data) / 3; % number of joints
        positions = reshape(data, [3, nJoints])';
        
        % Clear the current frame
        cla;
        
        % Plot points for each joint without connecting lines
        scatter3(positions(:,1), positions(:,2), positions(:,3), 'filled', 'MarkerFaceColor', 'b');
        
        % Set axes limits
        set(gca, 'XLim', [-axLimit axLimit], 'YLim', [-axLimit axLimit], 'ZLim', zlim);
        
        % Update the plot
        drawnow;
    end
end

% Close the file
fclose(fileID);
close all;

% At this point, dataMatrix contains all the data from the file
% Assuming dataMatrix contains your data from the file
% Plot the 21st column versus the last column

% Assuming dataMatrix contains your data from the file
% Get the number of rows in dataMatrix
numRows = size(dataMatrix, 1);

% Create a vector of row indices
rowIndices = 1:numRows;

% % Plot the 21st column against the row indices
% figure;
% subplot(2,1,1);  % Create a subplot for the 21st column
% plot(rowIndices, dataMatrix(:, 21), 'o-');
% xlabel('Row Index');
% ylabel('21st Column');
% title('Plot of 21st Column vs Row Index');
% grid on;
% 
% % Plot the last column against the row indices
% subplot(2,1,2);  % Create a subplot for the last column
% plot(rowIndices, dataMatrix(:, end), 'o-');
% xlabel('Row Index');
% ylabel('Last Column');
% title('Plot of Last Column vs Row Index');
% grid on;
