% Load the XML file
xmlFileNames = ['clips/annotations_1.xml';'clips/annotations_2.xml'];
imgDir = 'clips/train_original/';
outDir = 'clips/train_expanded/';
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
imgcount = 0;
expand_factor = 10;
for l = 1:2
    xmlDoc = xmlread(xmlFileNames(l,:));
    imageNodes = xmlDoc.getElementsByTagName('image');
    % Loop over each image in XML
    for i = 0:imageNodes.getLength-1
        % Get the image node
        imageNode = imageNodes.item(i);
        
        % Get the image name
        imageName = char(imageNode.getAttribute('name'));
        img = imread([imgDir,imageName]);
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
        end
        imwrite(img,[outDir,num2str(imgcount),'.jpg']) %Output the orginal image along with annotations
        imageData.lanes = {lanes(1,:), lanes(2,:), lanes(3,:), lanes(4,:)};  % Lists for lanes
        imageData.h_samples = h_samples;     % Assign predefined h_samples
        imageData.raw_file = strcat(outDir, num2str(imgcount),'.jpg');
        imgcount = imgcount + 1;
        jsonStr = jsonencode(imageData);
        if rem(i,10) == 0
            fprintf(fileID, '%s\n', jsonStr);
        else
            fprintf(fileID2, '%s\n', jsonStr);
        end
        img = im2double(img);
        for j=1:expand_factor %Output augmented images according to expansion factor
            flip_flag = round(rand(1));
            if flip_flag
                img = flip(img,2);
                imageData.lanes = {1280-lanes(1,:), 1280-lanes(2,:), 1280-lanes(3,:), 1280-lanes(4,:)};
                imageData.lanes{1}(imageData.lanes{1}==1282) = -2;% Set empty back to empty
                imageData.lanes{2}(imageData.lanes{2}==1282) = -2;
                imageData.lanes{3}(imageData.lanes{3}==1282) = -2;
                imageData.lanes{4}(imageData.lanes{4}==1282) = -2;
            else
                imageData.lanes = {lanes(1,:), lanes(2,:), lanes(3,:), lanes(4,:)};  % Lists for lanes
            end
            imageData.h_samples = h_samples;     % Assign predefined h_samples
            imageData.raw_file = strcat(outDir, num2str(imgcount),'.jpg');
            jsonStr = jsonencode(imageData);
            % Convert RGB image to HSV for hue and saturation adjustments
            hsvImg = rgb2hsv(img);
            % Adjust Hue (H)
            hsvImg(:,:,1) = hsvImg(:,:,1) + 0.2*rand(1)-0.1;
            hsvImg(:,:,1) = mod(hsvImg(:,:,1), 1);
            % Adjust Saturation (S)
            hsvImg(:,:,2) = hsvImg(:,:,2) * (0.85+0.3*rand(1));
            hsvImg(:,:,2) = min(hsvImg(:,:,2), 1);
            % Convert back to RGB after modifying H and S
            adjustedImg = hsv2rgb(hsvImg);
            % Adjust Brightness and Contrast using imadjust
            adjustedImg = imadjust(adjustedImg, [], [], (0.8+0.4*rand(1))); % 1.2 gamma increases brightness
            % Append the data for this image to the main JSON array
            imwrite(adjustedImg,[outDir,num2str(imgcount),'.jpg'])
            imgcount = imgcount + 1;
            if rem(i,10) == 0
                fprintf(fileID, '%s\n', jsonStr);
            else
                fprintf(fileID2, '%s\n', jsonStr);
            end
        end   
    end
end
fclose(fileID);
fclose(fileID2);
disp('JSON file has been generated.');