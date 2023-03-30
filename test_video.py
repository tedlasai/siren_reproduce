from siren import mySiren
from data import Video
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import argparse
import os
import numpy as np
import torch
from PIL import Image


def test(video_num, device, chkpoint):
    epochs=100000 #number used in paper for video training

    model = mySiren(in_size=3, out_size=3, hidden_layers=3, hidden_size=1024)
    model.to(device=device)
    video = Video(video_num=video_num)

    checkpoint = torch.load(chkpoint)
    model.load_state_dict(checkpoint)

    with torch.no_grad():
        num_splits = video.num_frames 
        split_size = video.coords.shape[0]//num_splits
        for i in range(num_splits): 
            coords_split = video.coords[i*split_size:(i+1)*split_size]
            video_split = video.vid[i*split_size:(i+1)*split_size]
            coords_split = coords_split.to(device)
            video_split = video_split.to(device)
            model_out = model(coords_split)
            model_out = (model_out+1)*0.5*255
            model_out = model_out.detach().cpu().numpy()
            model_out = np.clip(model_out, 0, 255)
            model_out = model_out.astype(np.uint8)
            model_out = model_out.reshape(video.original_shape[1], video.original_shape[2], video.original_shape[3])
            im = Image.fromarray(model_out)
            im.save(f"video_frames/video{video_num}_frame{i}.jpg")
                        



def get_args():
    parser = argparse.ArgumentParser(description='Train video network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-v', '--video-num', metavar='VN', type=int, nargs='?', default=1,
    help='Learning rate', dest='video_num')
    parser.add_argument('-c', '--checkpoint', dest='chkpoint', type=str, default="checkpoints_video/epoch100000.pth",
                    help='Number of epochs to save a checkpoint')
    return parser.parse_args()

            

if __name__ == '__main__':
    
    #get free gpu
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Testing of SIRENs for Video Compression')
    args = get_args()
    def get_freer_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)

    gpu_num = get_freer_gpu()
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    test(device=device, chkpoint=args.chkpoint,video_num = args.video_num)