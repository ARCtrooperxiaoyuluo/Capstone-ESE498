% Load the XML file
xmlFileNames = ['clips/annotations_1.xml';'clips/annotations_2.xml'];
imgDirs = ['clips/train_original/'];
fileID = fopen('train_tasks_val.json', 'w');
fileID2 = fopen('train_tasks_train.json', 'w');
jsonData_train = '';
jsonData_val = '';
h_samples = [180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, ...
             360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, ...
             480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, ...
             600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710];

if fileID == -1 || fileID2 == -1
    error('Error opening the file for writing.');
end
for l = 1:2
    xmlDoc = xmlread(xmlFileNames(l,:));
    imageNodes = xmlDoc.getElementsByTagName('image');
    imgDir = imgDirs(1,:);
    % Loop over each image in XML
    for i = 0:imageNodes.getLength-1
        % Get the image node
        imageNode = imageNodes.item(i);
        
        % Get the image name
        imageName = char(imageNode.getAttribute('name'));
        fprintf('Image Name: %s\n', imageName);
        
        % Get all polyline elements for this image
        polylineNodes = imageNode.getElementsByTagName('polyline');
        
        % Loop over each polyline
        lanes = zeros([4,length(h_samples)]);
        lanes = lanes -2;
        for j = 0:polylineNodes.getLength-1
            % Get the polyline node
            polylineNode = polylineNodes.item(j);
            
            % Get the label
            label = char(polylineNode.getAttribute('label'));
            
            % Get the points
            points = char(polylineNode.getAttribute('points'));
            coord = regexp(points, '[\d.]+', 'match');
            for k = 4:2:length(coord)
                slope = (str2double(coord{k-2})-str2double(coord{k}))/(str2double(coord{k-3})-str2double(coord{k-1}));
                for h=180:10:710
                    if h<min(str2double(coord{k-2}), str2double(coord{k})) || h>max(str2double(coord{k-2}), str2double(coord{k}))
                        continue;
                    else
                        lanes(j+1,(h-170)/10) = round((h-str2double(coord{k-2}))/slope+str2double(coord{k-3}));
                    end
                end
            end
            % Print the label and corresponding points
            fprintf('  Label: %s, Points: %s\n', label, points);
        end
        imageData.lanes = {lanes(1,:), lanes(2,:), lanes(3,:), lanes(4,:)};  % Lists for lanes
        imageData.h_samples = h_samples;     % Assign predefined h_samples
        
        % Adjust the raw_file path to match the desired format
        imageData.raw_file = strcat(imgDir, imageName);
        jsonStr = jsonencode(imageData);
        % Append the data for this image to the main JSON array
        if rem(i,10) == 0
            fprintf(fileID, '%s\n', jsonStr);
        else
            fprintf(fileID2, '%s\n', jsonStr);
        end
    end
end
fclose(fileID);
fclose(fileID2);
disp('JSON file has been generated.');