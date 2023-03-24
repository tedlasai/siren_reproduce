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
import math
import matplotlib.colors as colors

#this funciton is borrowwed from the authors
def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

#this funciton is borrowed from the authros
def grads2img(gradients):
    mG = gradients.detach().squeeze(0).permute(-2, -1, -3).cpu()
    nRows = mG.shape[0]
    nCols = mG.shape[1]
    mGr = mG[:, :, 0]
    mGc = mG[:, :, 1]
    mGa = np.arctan2(mGc, mGr)
    mGm = np.hypot(mGc, mGr)
    mGhsv = np.zeros((nRows, nCols, 3), dtype=np.float32)
    mGhsv[:, :, 0] = (mGa + math.pi) / (2. * math.pi)
    mGhsv[:, :, 1] = 1.

    nPerMin = np.percentile(mGm, 5)
    nPerMax = np.percentile(mGm, 95)
    mGm = (mGm - nPerMin) / (nPerMax - nPerMin)
    mGm = np.clip(mGm, 0, 1)

    mGhsv[:, :, 2] = mGm
    mGrgb = colors.hsv_to_rgb(mGhsv)
    return torch.from_numpy(mGrgb).permute(2, 0, 1)

def train(lr, device, chkpointperiod):
    epochs=10000 #number used in paper for audio training

    model = mySiren(in_size=2, out_size=1, hidden_layers=3, hidden_size=256)
    model.to(device=device)
    poisson = Poisson()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(poisson, batch_size=1, pin_memory=True, num_workers=0)
    for epoch in range(epochs):
        for coord_values, gradient_gt, laplace_gt in dataloader:
            dir_checkpoint = f'./checkpoints_poisson/'
            coord_values, gradient_gt, laplace_gt = coord_values.to(device), gradient_gt.to(device), laplace_values.to(device)
            coord_values = coord_values.requires_grad_(True)
            model_out = model(coord_values)
            grad_outputs = torch.ones_like(model_out)
            gradient = torch.autograd.grad(model_out, [coord_values], grad_outputs=grad_outputs, create_graph=True)[0] #first derviative

            laplace = torch.autograd.grad(model_out, [coord_values], grad_outputs=grad_outputs, create_graph=True)[0] 

            mse = nn.MSELoss()

            loss = mse(gradient, gradient_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss:{loss} ")


            out_grad = grads2img(lin2img(gradient))
            gt_grad = grads2img(lin2img(poisson_values))
            
            mse_grad = torch.mean((out_grad-gt_grad)**2)
            psnr_grad = 10*torch.log10(1/mse_grad) #use 4 because range from 0- 2

            def normalize(input):
                max = torch.max(input)
                min = torch.min(input)

                return (input - min)/(max-min)


            out_im = normalize(model_out) 
            gt_im = normalize(poisson.image.to(device))
            
            mse_im = torch.mean((out_im-gt_im)**2)
            psnr_im = 10*torch.log10(1/mse_im) #use 4 because range from 0- 2

            wandb.log({"Loss": loss, "PSNR_grad": psnr_grad, "PSNR_im": psnr_im},) 


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
    return parser.parse_args()

            

if __name__ == '__main__':
    
    #get free gpu
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of SIRENs for Poisson Reconstruction')
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
        "reproduction_task": "poisson"
        }
    
    wandb.init(project="reproduction",
    config = wandb.config,
    notes="",
    tags=["baseline"])

    train(lr=args.lr, device=device, chkpointperiod=args.chkpointperiod)