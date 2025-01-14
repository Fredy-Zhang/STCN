import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class GenericTestDataset(Dataset):
    def __init__(self, data_root, res=480):
        #self.image_dir = path.join(data_root, 'JPEGImages')
        #self.mask_dir = path.join(data_root, 'Annotations')
        self.data_root = path.join(data_root, 'annotations')
        self.val_path = path.join(data_root, 'val.txt')

        self.videos = []
        self.shape = {}
        self.frames = {}

        vid_list = []
        if self.val_path is not None:
            with open(self.val_path, "r") as f:
                while True:
                    file_name = f.readline()
                    if not file_name:
                        break
                    file_name = file_name.strip("\n")
                    vid_list.append(file_name)

        ### vid_list = sorted(os.listdir(self.image_dir))
        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.data_root, vid, 'rgb')))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_mask = os.listdir(path.join(self.data_root, vid, 'mask'))[0]
            _mask = np.array(Image.open(path.join(self.data_root, vid, 'mask', first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

        if res != -1:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(res, interpolation=InterpolationMode.BICUBIC),
            ])

            self.mask_transform = transforms.Compose([
                transforms.Resize(res, interpolation=InterpolationMode.NEAREST),
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])

            self.mask_transform = transforms.Compose([
            ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video
        info['frames'] = self.frames[video] 
        info['size'] = self.shape[video] # Real sizes
        info['gt_obj'] = {} # Frames with labelled objects

        #vid_im_path = path.join(self.image_dir, video)
        #vid_gt_path = path.join(self.mask_dir, video)
		
        vid_im_path = path.join(self.data_root, video, 'rgb')
        vid_gt_path = path.join(self.data_root, video, 'mask')

        frames = self.frames[video]

        images = []
        masks = []
        masks_num = 0
        for i, f in enumerate(frames):
            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))
            
            mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
            if path.exists(mask_file):
                mask = Image.open(mask_file).convert('P')
                palette = mask.getpalette()
                masks.append(np.array(mask, dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
                masks_num = masks_num + 1
            else:
                # Mask not exists -> nothing in it
                masks.append(np.zeros(self.shape[video]))
        
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        
        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        # Resize to 480p
        masks_tmp = masks
        masks_tmp = masks_tmp.unsqueeze(2)
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
            'palette': np.array(palette),
            'gts': masks_tmp,
            'msk_n':masks_num,
        }

        return data

    def __len__(self):
        return len(self.videos)