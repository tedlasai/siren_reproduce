from models.siren import mySiren
from dataloaders.dataio import CelebA
from torch import nn, optim
from torch.utils.data import DataLoader
import logging
import argparse
import os
import numpy as np
import torch
import wandb
from models.meta_siren import myMetaSiren
from models.encoder import myEncoder
from models.hypernet import myHypernet

def train(lr, device, chkpointperiod):
    epochs=175 #number used in paper for video training

    model = myMetaSiren(in_size=2, out_size=3, hidden_layers=3, hidden_size=256)
    encoder = myEncoder()
    hypernet = myHypernet(model.meta_named_parameters())
    model.to(device=device)
    encoder.to(device=device)
    hypernet.to(device=device)
    print(model)

    celeba_dataset = CelebA(split='train')
    all_params = list(encoder.parameters()) + list(hypernet.parameters())
    optimizer = optim.Adam(all_params, lr=lr)
    dataloader = DataLoader(celeba_dataset, batch_size=200, pin_memory=True, num_workers=0, shuffle=True)


    for epoch in range(epochs):
        for coord_values, sparse_ims, gt_ims in dataloader:
            dir_checkpoint = f'./checkpoints_reconstruction/'
            coord_values, sparse_ims, gt_ims = coord_values.to(device), sparse_ims.to(device), gt_ims.to(device)
            encoder_out = encoder(sparse_ims)
            siren_params, weights_total = hypernet(encoder_out)
            model_out = model(coord_values, siren_params)
            model_out = torch.moveaxis(model_out, (1), (2))
            model_out = model_out.reshape((model_out.shape[0], model_out.shape[1],32,32))
            mse = nn.MSELoss()

        

            loss_im = mse(model_out, gt_ims)*3 #loss_im is only divided by h*w not H*w*c #thus I multiply the three back in
            print("ENCODER OOUT SHAPPE", encoder_out.shape)
            loss_latent = torch.mean(encoder_out**2)*0.1
            loss_weights = weights_total*100
            loss = loss_im + loss_latent + loss_weights

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss:{loss} ")

            psnr = 10*torch.log10(4/(loss_im/3))
            wandb.log({"Loss_im": loss_im/3, "Loss_latent": loss_latent, "loss_weights": loss_weights, "Loss_total": loss, "PSNR": psnr, "Epoch": epoch},) #psnr for small batch


        if (epoch + 1) % chkpointperiod == 0 or epoch==(epochs-1):#for last epoch output tthe full psnr
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            states = {"encoder": encoder.state_dict(), "hypernet":hypernet.state_dict()}

            torch.save(states, dir_checkpoint + f'epoch{epoch + 1}.pth')
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
    parser = argparse.ArgumentParser(description='Train reconstruction network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00005,
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
