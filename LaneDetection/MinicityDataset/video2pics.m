% Define the video file and output directory
videoFiles = ['video2.mp4';'video3.mp4';'video4.mp4';'video5.mp4'];
videoFiles = videoFiles';
outdir = [pwd, '/clips/2/'];

if ~exist(outdir, 'dir')
    mkdir(outdir);
end
frameInterval = 10;
frameCount = 0260;
% Create a VideoReader object to read the video
for videoFile = videoFiles
    vidObj = VideoReader(videoFile');
    
    % Set frame extraction interval (every 10th frame)

    
    % Loop through the video frames
    while hasFrame(vidObj)
        frameCount = frameCount + 1;
        % Read the current frame
        frame = readFrame(vidObj);
        
        % Save the frame every 10 frames
        if mod(frameCount, frameInterval) == 0
            % Create a filename for the frame
            frameFilename = fullfile(outdir, sprintf('%04d.jpg', frameCount));
            % Resize the frame to 1280x720 resolution
            resizedFrame = imresize(frame, [720, 1280]);
            % Write the frame to the file
            rotatedFrame = imrotate(resizedFrame, 180);
            imwrite(rotatedFrame, frameFilename);
        end
    end
end
disp('Frame extraction completed.');

