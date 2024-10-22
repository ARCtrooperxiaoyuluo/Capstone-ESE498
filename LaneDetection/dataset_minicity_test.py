import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import cv2
import ujson as json
from model import utils


VGG_MEAN = [103.939, 116.779, 123.68]


class MinicityDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_dir, size=(512,288), transform=None):
        """
        Args:
            dataset_dir: The directory path of the dataset
            phase: 'train', 'val', or 'test'
        """
        self.dataset_dir = dataset_dir
        self.size = size
        self.transform = transform
        assert os.path.exists(dataset_dir), 'Directory {} does not exist!'.format(dataset_dir)
        task_file = os.path.join(dataset_dir, 'test_tasks.json')
        try:
            self.image_list = [json.loads(line)['raw_file'] for line in open(task_file).readlines()]
        except BaseException:
            raise Exception(f'Fail to load {task_file}.')

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        '''OpenCV'''
        img_path = os.path.join(self.dataset_dir, self.image_list[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image -= VGG_MEAN
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255

        clip, seq, frame = self.image_list[idx].split('/')[-3:]
        path = '/'.join([clip, seq, frame])

        sample = {'input_tensor': image, 'raw_file':self.image_list[idx], 'path':path}

        return sample




if __name__ == '__main__':
    
    test_set = MinicityDataset('/root/Projects/lane_detection/dataset/tusimple/test_set')

    # train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    # valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True, num_workers=1)
