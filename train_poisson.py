from siren import mySiren
from data import Poisson
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import argparse
import os
import numpy as np
import torch
import wandb
import matplotlib.colors as colors
from utils import lin2img, grads2img, rescale_img
import cv2
import cmapy

def train(lr, device, chkpointperiod, gradientlaplace):
    epochs=10000 #number used in paper for audio training

    model = mySiren(in_size=2, out_size=1, hidden_layers=3, hidden_size=256)
    model.to(device=device)
    if(gradientlaplace == 0):
        imageMult = 10
        num_items=0
    else:
        imageMult = 10000
        num_items=0
    poisson = Poisson(imageMult, num_items=num_items)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(poisson, batch_size=1, pin_memory=True, num_workers=0)
    for epoch in range(epochs):
        for coord_values, image_gt, gradient_gt, laplace_gt in dataloader:
            dir_checkpoint = f'./checkpoints_poisson/'
            coord_values, image_gt, gradient_gt, laplace_gt = coord_values.to(device), image_gt.to(device), gradient_gt.to(device), laplace_gt.to(device)
            coord_values = coord_values.requires_grad_(True)
            model_out = model(coord_values)
            gradient = torch.autograd.grad(model_out, [coord_values], grad_outputs=torch.ones_like(model_out), create_graph=True)[0] #first derviative
         

            laplace_total = 0.
            for i in range(2):
                laplace_total += torch.autograd.grad(gradient[:,:,i], coord_values, torch.ones_like(gradient[:,:,i]), create_graph=True)[0][:,:,i:i+1]

            

            mse = nn.MSELoss()

            if gradientlaplace == 0: #supervising training with gradient

                loss = mse(gradient, gradient_gt)
            else:
                loss = mse(laplace_total, laplace_gt)

            loss.backward(retain_graph=False)
            optimizer.step()
            optimizer.zero_grad()
            print(f"Loss:{loss} ")


            out_grad = grads2img(lin2img(gradient))
            gt_grad = grads2img(lin2img(gradient_gt))
            
            mse_grad = torch.mean((out_grad-gt_grad)**2)
            psnr_grad = 10*torch.log10(1/mse_grad) 


            out_im = rescale_img(model_out, mode='scale', perc=1)
            out_im = (out_im*255).detach().cpu().numpy().astype(np.uint8)/255.0
            gt_im = rescale_img(image_gt, mode='scale', perc=1)
            gt_im = (gt_im*255).detach().cpu().numpy().astype(np.uint8)/255.0


            mse_im = np.mean((out_im-gt_im)**2) #divide by 255 because uint8 scale needs to be removed
            psnr_im = 10*np.log10(1/mse_im) #use 4 because range from 0- 2

            out_laplace = rescale_img(laplace_total, mode='scale', perc=1)
            out_laplace = (out_laplace.squeeze()*255).detach().cpu().numpy().astype(np.uint8)/255.0
            gt_laplace = rescale_img(laplace_gt, mode='scale', perc=1)
            gt_laplace = (gt_laplace.squeeze()*255).detach().cpu().numpy().astype(np.uint8)/255.0

            mse_laplace = np.mean((out_laplace-gt_laplace)**2) #divide by 255 because uint8 scale needs to be removed
            psnr_laplace = 10*np.log10(1/mse_laplace) #use 4 because range from 0- 2

            wandb.log({"Loss": loss, "PSNR_grad": psnr_grad, "PSNR_im": psnr_im, "PSNR_laplace": psnr_laplace},) 


            if (epoch + 1) % chkpointperiod == 0 or epoch==(epochs-1):
                if not os.path.exists(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')

                torch.save(model.state_dict(), dir_checkpoint + f'epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved!')




def get_args():
    parser = argparse.ArgumentParser(description='Train video network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=5,
                    help='Number of epochs to save a checkpoint')
    parser.add_argument('-gl', '--gradient-or-laplace', dest='gradientlaplace', type=int, default=0,
                    help='Supervise training with graidnet or laplaician')
    return parser.parse_args()

            

if __name__ == '__main__':
    
    #get free gpu
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of SIRENs for Poisson Reconstruction')
    args = get_args()
    def get_freer_gpu():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        return np.argmin(memory_available)

    gpu_num = get_freer_gpu()
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    wandb.config = {
        "learning_rate": args.lr,
        "reproduction_task": "poisson"
        }
    
    wandb.init(project="reproduction",
    config = wandb.config,
    notes="",
    tags=["baseline"])

    train(lr=args.lr, device=device, chkpointperiod=args.chkpointperiod, gradientlaplace = args.gradientlaplace)