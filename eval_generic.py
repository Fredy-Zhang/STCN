"""
Generic evaluation script 
The segmentation mask for each object when they first appear is required
(YouTubeVOS style, but dense)

Optimized for compatibility, not speed.
We will resize the input video to 480p -- check generic_test_dataset.py if you want to change this behavior
AMP default on.

Usage: python eval_generic.py --data_path <path to data_root> --output <some output path>

Data format:
    data_root/
        JPEGImages/
            video1/
                00000.jpg
                00001.jpg
                ...
            video2/
                ...
        Annotations/
            video1/
                00000.png
            video2/
                00000.png
            ...
"""


import os
from os import path
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from model.eval_network import STCN
from dataset.generic_test_dataset import GenericTestDataset
from util.tensor_util import unpad, compute_tensor_iou
from inference_core import InferenceCore

from progressbar import progressbar

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='saves/stcn.pth')
parser.add_argument('--data_path')
parser.add_argument('--output')
parser.add_argument('--top', type=int, default=20)
parser.add_argument('--amp_off', action='store_true')
parser.add_argument('--mem_every', default=5, type=int)
parser.add_argument('--include_last', help='include last frame as temporary memory?', action='store_true')
args = parser.parse_args()

data_path = args.data_path
out_path = args.output
args.amp = not args.amp_off

# Simple setup
os.makedirs(out_path, exist_ok=True)
torch.autograd.set_grad_enabled(False)

# Setup Dataset
test_dataset = GenericTestDataset(data_root=data_path)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# Load our checkpoint
top_k = args.top
prop_model = STCN().cuda().eval()

# Performs input mapping such that stage 0 model can be loaded
prop_saved = torch.load(args.model)
for k in list(prop_saved.keys()):
    if k == 'value_encoder.conv1.weight':
        if prop_saved[k].shape[1] == 4:
            pads = torch.zeros((64,1,7,7), device=prop_saved[k].device)
            prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
prop_model.load_state_dict(prop_saved)

# Start eval
for data in progressbar(test_loader, max_value=len(test_loader), redirect_stdout=False):

    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb']
        msk = data['gt'][0]
        info = data['info']
        name = info['name'][0]
        num_objects = len(info['labels'][0])
        gt_obj = info['gt_obj']
        size = info['size']
        palette = data['palette'][0]
        meanIoU = []

        print('Processing', name, '...')
        print("mask num: ", data["msk_n"])
        print("msk shape: ", msk.shape)
        print("rgb shape: ", rgb.shape)

        # Frames with labels, but they are not exhaustively labeled
        frames_with_gt = sorted(list(gt_obj.keys()))
        processor = InferenceCore(prop_model, rgb, num_objects=num_objects, top_k=top_k, 
                                    mem_every=args.mem_every, include_last=args.include_last)

        processor.interact(msk[:,0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size (we made it 480p)
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        
        print("processor t: ", processor.t)
        for ti in range(processor.t):
            #print("No. ", ti)
            prob = unpad(processor.prob[:,ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)
            meanIoU.append(compute_tensor_iou(torch.flatten(out_masks[ti][0,:,:]).detach().cpu().int(), torch.flatten(data['gts'][0,0,ti,0,:,:]).int()))

        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        # Remap the indices to the original domain
        idx_masks = np.zeros_like(out_masks)
        for i in range(1, num_objects+1):
            backward_idx = info['label_backward'][i].item()
            idx_masks[out_masks==i] = backward_idx
        
        # Save the results
        this_out_path = path.join(out_path, name)
        os.makedirs(this_out_path, exist_ok=True)
        for f in range(idx_masks.shape[0]):
            img_E = Image.fromarray(idx_masks[f])
            img_E.putpalette(palette)
            img_E.save(os.path.join(this_out_path, info['frames'][f][0].replace('.jpg','.png')))
        print("Mean IOU is: ", sum(meanIoU)/len(meanIoU), "in ", this_out_path)

        del rgb
        del msk
        del processor
