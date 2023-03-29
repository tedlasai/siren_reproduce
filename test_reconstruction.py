from siren import mySiren
from dataio import CelebA
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import argparse
import os
import numpy as np
import torch
import wandb
from meta_siren import myMetaSiren
from encoder import myEncoder
from hypernet import myHypernet

def test(device, chkpoint):

    model = myMetaSiren(in_size=2, out_size=3, hidden_layers=3, hidden_size=256)
    encoder = myEncoder()
    hypernet = myHypernet(model.meta_named_parameters())
    model.to(device=device)
    encoder.to(device=device)
    hypernet.to(device=device)
    celeba_dataset = CelebA(split='test')
    dataloader = DataLoader(celeba_dataset, batch_size=1, pin_memory=True, num_workers=0)


    for coord_values, sparse_ims, gt_ims in dataloader:
        dir_checkpoint = f'./checkpoints_reconstruction/'
        coord_values, sparse_ims, gt_ims = coord_values.to(device), sparse_ims.to(device), gt_ims.to(device)
        encoder_out = encoder(sparse_ims)
        siren_params, weights_total = hypernet(encoder_out)
        model_out = model(coord_values, siren_params)
        model_out = torch.moveaxis(model_out, (1), (2))
        model_out = model_out.reshape((model_out.shape[0], model_out.shape[1],32,32))
    



def get_args():
    parser = argparse.ArgumentParser(description='Train reconstruction network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-c', '--checkpoint', dest='chpoint', type=str, default="checkpoints_reconstruction/epoch35.pth",
                    help='Number of epochs to save a checkpoint')
    return parser.parse_args()

            

if __name__ == '__main__':
    
    #get free gpu
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of SIRENs for Reconstruction')
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
        "reproduction_task": "reconstruction"
        }
    
    wandb.init(project="reconstruction",
    config = wandb.config,
    notes="",
    tags=["baseline"])

    test(device=device, chkpointperiod=args.chkpointperiod)
