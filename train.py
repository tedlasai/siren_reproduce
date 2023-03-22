from siren import mySiren
from data import Video
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import argparse
import os
import numpy as np
import torch
import wandb

def get_args():
    parser = argparse.ArgumentParser(description='Train video network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    return parser.parse_args()

def train(lr, device ):
    epochs=100000 #number used in paper for video training

    model = mySiren(in_size=2, out_size=3, hidden_layers=5, hidden_size=1024)
    model.to(device=device)
    video = Video(video_num=1, num_items=10)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(video, batch_size=1, pin_memory=True, num_workers=0)
    for epoch in range(epochs):
        print("Epoch")
        for coord_values, vid_values in dataloader:
            coord_values, vid_values = coord_values.to(device), vid_values.to(device)
            model_out = model(coord_values)
            mse = nn.MSELoss()

            loss = mse(model_out, vid_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss:{loss} ")
        





            

if __name__ == '__main__':
    
    #get free gpu
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of SIRENs for Video Compression')
    args = get_args()
    def get_freer_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmax(memory_available)

    gpu_num = get_freer_gpu()
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    wandb.config = {
        "learning_rate": args.lr,
        "reproduction_task": "video"
        }
    
    wandb.init(project="reproduction",
    config = wandb.config,
    notes="",
    tags=["baseline"])

    train(lr=args.lr, device=device)