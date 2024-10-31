To run the model, a checkpoint file, which includes all the weights, needs to be included. The checkpoint file is too large for Github, so there's a link to Baidu Drive: https://pan.baidu.com/s/1LrvIVC_fdTAlYcRoEAKNLQ#list/path=%2F (提取码: u6wa) I trained another checkpoint which works better: https://drive.google.com/file/d/1Q6AFOM_yUyVG2X6rx1_UMBfj_lQmSWes/view?usp=sharing

This package is copied and modified based on this repo: https://github.com/ShenhanQian/Lane_Detection-An_Instance_Segmentation_Approach/blob/master/README.md.
The main modifications eliminate the need for the CUDA and TuSimple datasets.

To test the model, first update test_tasks.json with createTestTasks.m by designating the data folder and number of frames to test.
Then use the following command:
python test_lanenet-tusimple_benchmark.py ^
        --data_dir D:\Courses\ESE498\InstanceSegmentation\MinicityDataset ^
        --arch fcn ^
        --ckpt_path D:\Courses\ESE498\InstanceSegmentation\check_point\ckpt_2024-10-29_02-25-30_None\ckpt_2024-10-29_02-25-30_epoch-600.pth ^
--show

To train a new model from scratch, update train_tasks_val.json and train_tasks_train.json with CVAT2TuSimple.m. Labeled data in CVAT format must be present. 
Command:
python train_lanenet.py --data_dir D:\Courses\ESE498\InstanceSegmentation\MinicityDataset --arch fcn --ckpt_path D:\Courses\ESE498\ckpt_FCN-Res18-1E1D-b1d0.01-v0.2d1_epoch-590.pth
