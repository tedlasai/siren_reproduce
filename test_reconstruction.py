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
from PIL import Image
def test(device, chkpoint):

    model = myMetaSiren(in_size=2, out_size=3, hidden_layers=3, hidden_size=256)
    encoder = myEncoder()
    hypernet = myHypernet(model.meta_named_parameters())
    model.to(device=device)
    encoder.to(device=device)
    hypernet.to(device=device)

    checkpoint = torch.load(chkpoint, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    hypernet.load_state_dict(checkpoint["hypernet"])
    del checkpoint
    

    scenarios = ["random_test", "random_test", "random_test", "half", "full"]
    ranges = [10, 100, 1000, -1, -1] # last two don't use rnage parameter anyways

    for i in range(5):
        celeba_dataset = CelebA(split='test', test_sparsity=scenarios[i], train_sparsity_range=ranges[i])
        dataloader = DataLoader(celeba_dataset, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)
        count = 0
        loss = 0
        for coord_values, sparse_ims, gt_ims in dataloader:
            coord_values, sparse_ims, gt_ims = coord_values.to(device), sparse_ims.to(device), gt_ims.to(device)
            encoder_out = encoder(sparse_ims)
            siren_params, weights_total = hypernet(encoder_out)
            model_out = model(coord_values, siren_params)
            model_out = torch.moveaxis(model_out, (1), (2))
            model_out = model_out.reshape((model_out.shape[0], model_out.shape[1],32,32))

            mse = nn.MSELoss()
            model_out = torch.clip((model_out+1)*0.5, 0, 1)
            gt_ims = torch.clip((gt_ims+1)*0.5, 0, 1)
            loss_im = mse(model_out, gt_ims)
            with torch.no_grad():
                loss += loss_im

                model_out = model_out*255
                model_out = torch.moveaxis(model_out.squeeze(), (0,1,2), (2,0,1))
                model_out = model_out.detach().cpu().numpy()
                model_out = np.clip(model_out, 0, 255)
                model_out = model_out.astype(np.uint8)
                model_out = model_out.reshape(32, 32, 3)
                im = Image.fromarray(model_out)
                im.save(f"reconstruction_frames/scenario_{i}/frame{count}_reconstruction.png")
        
                gt_ims = gt_ims*255
                gt_ims = torch.moveaxis(gt_ims.squeeze(), (0,1,2), (2,0,1))
                gt_ims = gt_ims.detach().cpu().numpy()
                gt_ims = np.clip(gt_ims, 0, 255)
                gt_ims = gt_ims.astype(np.uint8)
                gt_ims = gt_ims.reshape(32, 32, 3)
                im = Image.fromarray(gt_ims)
                im.save(f"reconstruction_frames/scenario_{i}/frame{count}_gt.png")

                
                sparse_ims = torch.moveaxis(sparse_ims.squeeze(), (0,1,2), (2,0,1))
                mask = torch.sum(sparse_ims, axis=2) ==0
                sparse_ims[mask, :] = 1
                sparse_ims = (sparse_ims+1)*0.5*255
                sparse_ims = sparse_ims.detach().cpu().numpy()
                sparse_ims = np.clip(sparse_ims, 0, 255)
                sparse_ims = sparse_ims.astype(np.uint8)
                sparse_ims = sparse_ims.reshape(32, 32, 3)
                
                im = Image.fromarray(sparse_ims)
                im.save(f"reconstruction_frames/scenario_{i}/frame{count}_sparse_initial.png")
                count+=1
        print(f"Scenario: {i} Loss: {loss/count}")



def get_args():
    parser = argparse.ArgumentParser(description='Train reconstruction network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-c', '--checkpoint', dest='chkpoint', type=str, default="checkpoints_reconstruction/epoch174.pth",
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

    test(device=device, chkpoint=args.chkpoint)
