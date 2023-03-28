from siren import mySiren
from data import Poisson
from torch.utils.data import DataLoader
import logging
import argparse
import os
import numpy as np
import torch
import wandb
from utils import lin2img, grads2img, rescale_img
from PIL import Image
import cmapy
import cv2

def test( device, chkpoint, gradientlaplace):
    model = mySiren(in_size=2, out_size=1, hidden_layers=3, hidden_size=256)
    model.to(device=device)
    if(gradientlaplace == 0):
        imageMult = 10
        num_items=0
    else:
        imageMult = 10000
        num_items=0
    poisson = Poisson(imageMult, num_items=num_items)
    dataloader = DataLoader(poisson, batch_size=1, pin_memory=True, num_workers=0)

    checkpoint = torch.load(chkpoint)
    model.load_state_dict(checkpoint)

    supervised_by =["gradient", "laplace"]

    for coord_values, image_gt, gradient_gt, laplace_gt in dataloader:
        dir_checkpoint = f'./checkpoints_poisson/'
        coord_values, image_gt, gradient_gt, laplace_gt = coord_values.to(device), image_gt.to(device), gradient_gt.to(device), laplace_gt.to(device)
        coord_values = coord_values.requires_grad_(True)
        model_out = model(coord_values)
        gradient = torch.autograd.grad(model_out, [coord_values], grad_outputs=torch.ones_like(model_out), create_graph=True)[0] #first derviative
        

        laplace_total = 0.
        laplace_total += torch.autograd.grad(gradient[:,:,0], coord_values, torch.ones_like(gradient[:,:,0]), create_graph=True)[0][:,:,0]
        laplace_total += torch.autograd.grad(gradient[:,:,1], coord_values, torch.ones_like(gradient[:,:,1]), create_graph=True)[0][:,:,1]
        laplace_total = laplace_total.squeeze()
        laplace_gt  = laplace_gt.squeeze()

    


        out_grad = grads2img(lin2img(gradient))
        out_grad = out_grad.moveaxis((0,1,2), (2,0,1))

        out_laplace = Image.fromarray((out_grad*255).detach().cpu().numpy().astype(np.uint8))
        out_laplace.save(f"poisson_out/gradient_supervised_by_{supervised_by[gradientlaplace]}.jpg")
        


        out_im = rescale_img(model_out, mode='scale', perc=1)
        out_im = (out_im.squeeze().reshape(poisson.original_shape[0], poisson.original_shape[1])*255).detach().cpu().numpy().astype(np.uint8)
        gt_im = rescale_img(image_gt, mode='scale', perc=1)
        gt_im = (gt_im*255).detach().cpu().numpy().astype(np.uint8)/255.0

        out_im = Image.fromarray(out_im)
        out_im.save(f"poisson_out/image_supervised_by_{supervised_by[gradientlaplace]}.jpg")

        out_laplace = rescale_img(laplace_total, mode='scale', perc=1)
        out_laplace = (out_laplace.squeeze()*255).detach().cpu().numpy().astype(np.uint8)

        gt_laplace = rescale_img(laplace_gt, mode='scale', perc=1)
        gt_laplace = (gt_laplace.squeeze()*255).detach().cpu().numpy().astype(np.uint8)/255.0

        out_laplace = out_laplace.reshape(poisson.original_shape[0], poisson.original_shape[1])
            
        
        out_laplace = cv2.cvtColor(cv2.applyColorMap(out_laplace, cmapy.cmap('RdBu')), cv2.COLOR_BGR2RGB)
        out_laplace = Image.fromarray(out_laplace)
        out_laplace.save(f"poisson_out/laplace_supervised_by_{supervised_by[gradientlaplace]}.jpg")



def get_args():
    parser = argparse.ArgumentParser(description='Train ppoisson network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-c', '--chkpoint', dest='chkpoint', type=str, default="checkpoints_poisson/epoch10000.pth",
                    help='checkpoint')
    parser.add_argument('-gl', '--gradient-or-laplace', dest='gradientlaplace', type=int, default=0,
                    help='Training was supervised with graidnet or laplaician')
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

    test(device=device, chkpoint=args.chkpoint, gradientlaplace = args.gradientlaplace)