% Define the file path
filename = '../data/raw/tocabi_data.txt';

% Open the file for reading
fileID = fopen(filename, 'r');

% Read data from the file
data = textscan(fileID, repmat('%f', 1, 91), 'Delimiter', ' ');
fclose(fileID); % Close the file

% Convert cell array to a matrix
dataMatrix = cell2mat(data);

% Scale the first column by 1/2000
dataMatrix(:, 1) = dataMatrix(:, 1) / 2000;

% Define the output file name (can be the same or different)
outputFilename = '../data/scaled/tocabi_data_scaled.txt';

% Open the file for writing
fileID = fopen(outputFilename, 'w');

% Write data back to the file
for i = 1:size(dataMatrix, 1)
    fprintf(fileID, '%f ', dataMatrix(i, :));
    fprintf(fileID, '\n');
end

% Close the file
fclose(fileID);
