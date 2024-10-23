To run the model, a checkpoint file, which includes all the weights, needs to be included. The checkpoint file is too large for Github, so there's a link to Baidu Drive: https://pan.baidu.com/s/1LrvIVC_fdTAlYcRoEAKNLQ#list/path=%2F (提取码: u6wa)
This package is copied and modified based on this repo: https://github.com/ShenhanQian/Lane_Detection-An_Instance_Segmentation_Approach/blob/master/README.md.
The main modifications eliminate the need for the CUDA and TuSimple datasets.

To test the model, first update test_tasks.json with createTestTasks.m by designating the data folder and number of frames to test.
Then use the following command:
python test_lanenet-tusimple_benchmark.py \
        --data_dir /Users/winston/Desktop/InstanceSegmentation/MinicityDataset \
        --arch fcn \
        --ckpt_path /Users/winston/Desktop/InstanceSegmentation/check_point/ckpt_2024-10-22_18-38-25_None/ckpt_2024-10-22_18-38-25_epoch-12.pth \
--show

To train a new model from scratch, update train_tasks_val.json and train_tasks_train.json with CVAT2TuSimple.m. Labeled data in CVAT format must be present. 
Command:
python train_lanenet.py \
        --data_dir /Users/winston/Desktop/InstanceSegmentation/MinicityDataset  \
        --arch fcn
