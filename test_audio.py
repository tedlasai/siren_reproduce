from siren import mySiren
from data import Audio
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import argparse
import os
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt

def test(audio_num, device, chkpoint):

    model = mySiren(in_size=1, out_size=1, hidden_layers=3, hidden_size=256)
    model.to(device=device)
    audio = Audio(audio_num=audio_num)
    dataloader = DataLoader(audio, batch_size=1, pin_memory=True, num_workers=0)


    checkpoint = torch.load(chkpoint)
    model.load_state_dict(checkpoint)

    for coord_values, audio_values in dataloader:
        coord_values, audio_values = coord_values.to(device), audio_values.to(device)
        model_out = model(coord_values)
        model_out = model_out
        plt.xticks([])
        plt.ylim(-1,1)
        plt.yticks(np.arange(-1, 2, step=1))

        plt.plot(coord_values.squeeze().cpu().detach().numpy(), model_out.squeeze().cpu().detach().numpy())
        plt.savefig(f"audio_vis/outplot_{audio_num}.jpg")




def get_args():
    parser = argparse.ArgumentParser(description='Train video network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-a', '--audio_num', metavar='AN', type=int, nargs='?', default=1,
    help='Learning rate', dest='audio_num')
    parser.add_argument('-c', '--chkpoint', dest='chkpoint', type=str, default="checkpoints_audio/epoch5000.pth",
                    help='Number of epochs to save a chkpoint')
    return parser.parse_args()

            

if __name__ == '__main__':
    
    #get free gpu
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of SIRENs for Audio Compression')
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
        "reproduction_task": "audio"
        }
    
    wandb.init(project="reproduction",
    config = wandb.config,
    notes="",
    tags=["baseline"])

    test( audio_num = args.audio_num, device=device, chkpoint=args.chkpoint)