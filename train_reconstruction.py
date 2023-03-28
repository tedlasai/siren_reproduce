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

def train(lr, device, chkpointperiod):
    epochs=100000 #number used in paper for video training

    model = myMetaSiren(in_size=3, out_size=3, hidden_layers=0, hidden_size=1024)
    encoder = myEncoder()
    hypernet = myHypernet(model.meta_named_parameters())
    model.to(device=device)
    encoder.to(device=device)
    hypernet.to(device=device)
    print(model)

    celeba_dataset = CelebA(split='train')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(celeba_dataset, batch_size=10, pin_memory=True, num_workers=0)


    for epoch in range(epochs):
        for coord_values, sparse_ims, gt_ims in dataloader:
            dir_checkpoint = f'./checkpoints_reconstruction/'
            coord_values, sparse_ims, gt_ims = coord_values.to(device), sparse_ims.to(device), gt_ims.to(device)
            encoder_out = encoder(sparse_ims)
            siren_params = hypernet(encoder_out)
            model_out = model(coord_values, siren_params)
            mse = nn.MSELoss()

            loss = mse(model_out, gt_ims)
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

                torch.save(model.state_dict(), dir_checkpoint + f'epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved!')

                with torch.no_grad():
                    pass

                    # expectation_psnr = torch.mean(psnr_values)
                    # psnr_variance = torch.var(psnr_values)
                   
                    # #expectation_psnr = 10*torch.log10(2/expectation_mse)

                    # #psnr_variance = 10*torch.log10(2/mse_variance)



                
                    # wandb.log({"PSNR_mean": expectation_psnr, "PSNR_variance": psnr_variance},) #for sirens this PS
                    # print("PSNR MEAN: ", expectation_psnr)
                    # print("PSNR VAR: ", psnr_variance)




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

    train(lr=args.lr, device=device, chkpointperiod=args.chkpointperiod)
