% Define the JSON file and output directory
jsonFile = 'test_tasks.json';

% Define frame numbers and the sample line structure
startFrame = 10;  % Starting frame number (e.g., 0010)
endFrame = 1980;  % Ending frame number (e.g., 4640)
frameStep = 10;   % Step size (based on frame extraction every 10 frames)
h_samples = 180:10:710;
lanes = [];
run_time = 1000;

% Open the JSON file for writing
fid = fopen(jsonFile, 'w');

% Loop through the frame numbers and create JSON lines
for frameNum = startFrame:frameStep:endFrame
    % Create the filename for the current frame (e.g., '0010.jpg', '0020.jpg', etc.)
    frameFilename = sprintf('%04d.jpg', frameNum);
    
    % Create the JSON structure
    jsonStruct.h_samples = h_samples;
    jsonStruct.lanes = lanes;
    jsonStruct.run_time = run_time;
    jsonStruct.raw_file = ['clips/train_original/', frameFilename]; % Use correct path format
    
    % Convert the structure to JSON format
    jsonLine = jsonencode(jsonStruct);
    
    % Write the JSON line to the file
    fprintf(fid, '%s\n', jsonLine);
end

% Close the JSON file
fclose(fid);

disp('JSON file generation completed.');
