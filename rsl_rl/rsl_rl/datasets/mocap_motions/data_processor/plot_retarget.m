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
zlim([8 10]);

% Set the same scaling for all axes
axis equal;
axis manual;  % Keep the axis limits fixed across frames
view(3);

% Initialize frame counter
frameCounter = 0;

% Animation loop
while ~feof(fileID)
    % Read one line of the file
    line = fgetl(fileID);
    
    % Increment frame counter
    frameCounter = frameCounter + 1;
    
    % Only plot every 1000th frame
    if mod(frameCounter, 10) == 1
        % Parse the line into an array of numbers
        data = str2double(strsplit(line));
        
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
