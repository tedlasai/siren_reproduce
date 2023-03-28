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

import pudb

def train(video_num, lr, device, chkpointperiod):
    epochs=100000 #number used in paper for video training

    model = mySiren(in_size=3, out_size=3, hidden_layers=3, hidden_size=1024)
    model.to(device=device)
    video = Video(video_num=video_num)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(video, batch_size=1, pin_memory=True, num_workers=0)
    print(model)
    for epoch in range(epochs):
        for coord_values, vid_values in dataloader:
            dir_checkpoint = f'./checkpoints_video/'
            coord_values, vid_values = coord_values.to(device), vid_values.to(device)
            model_out = model(coord_values)
            mse = nn.MSELoss()

            loss = mse(model_out, vid_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss:{loss} ")

            psnr = 10*torch.log10(4/loss)
            wandb.log({"Mse": loss, "PSNR_batch": psnr},) #psnr for small batch


            if (epoch + 1) % chkpointperiod == 0 or epoch==(epochs-1):#for last epoch output tthe full psnr
                if not os.path.exists(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')

                torch.save(model.state_dict(), dir_checkpoint + f'video_{video_num}_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved!')

                with torch.no_grad():
                    num_splits = video.num_frames #this is the number to split the data for evaluation so my GPU doesn't freak out about memory
                    split_size = video.coords.shape[0]//num_splits
                    psnr_values = torch.empty(num_splits)
                    for i in range(num_splits): #PSNR calculated across full video
                        coords_split = video.coords[i*split_size:(i+1)*split_size]
                        video_split = video.vid[i*split_size:(i+1)*split_size]
                        coords_split = coords_split.to(device)
                        #print("COORDS", coords_split)
                        video_split = video_split.to(device)
                        model_out = model(coords_split)
                        model_out = torch.clip(model_out, -1, 1)#clip output to make sure it stays in range
                        
                        mse_errors = mse(model_out,video_split)
                        eps=1e-10
                        psnr_values[i] = 10*torch.log10(4/(mse_errors+eps))
                        print("ITER: ", i)

                    expectation_psnr = torch.mean(psnr_values)
                    psnr_variance = torch.var(psnr_values)
                
                    wandb.log({"PSNR_mean": expectation_psnr, "PSNR_variance": psnr_variance},) 



def get_args():
    parser = argparse.ArgumentParser(description='Train video network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-v', '--video-num', metavar='VN', type=int, nargs='?', default=1,
    help='Learning rate', dest='video_num')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=5,
                    help='Number of epochs to save a checkpoint')
    return parser.parse_args()

            

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

    train(lr=args.lr, device=device, chkpointperiod=args.chkpointperiod,video_num = args.video_num)