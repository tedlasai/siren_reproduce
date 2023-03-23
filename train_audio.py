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

def train(audio_num, lr, device, chkpointperiod):
    epochs=5000 #number used in paper for audio training

    model = mySiren(in_size=1, out_size=1, hidden_layers=3, hidden_size=256)
    model.to(device=device)
    audio = Audio(audio_num=audio_num)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataloader = DataLoader(audio, batch_size=1, pin_memory=True, num_workers=0)
    for epoch in range(epochs):
        for coord_values, audio_values in dataloader:
            dir_checkpoint = f'./checkpoints_audio/'
            coord_values, audio_values = coord_values.to(device), audio_values.to(device)
            model_out = model(coord_values)
            mse = nn.MSELoss()

            loss = mse(model_out, audio_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Loss:{loss} ")

            psnr = 10*torch.log10(1/loss)
            
            wandb.log({"Mse": loss, "PSNR": psnr},) 


            if (epoch + 1) % chkpointperiod == 0 or epoch==(epochs-1):
                if not os.path.exists(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')

                torch.save(model.state_dict(), dir_checkpoint + f'epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved!')

                #no need to redo computation of model out because it is alreayd over the whole batch
                with torch.no_grad():
                    errors = (model_out-audio_values)**2
                    print("MSE Var: ", torch.var(errors))
                    print("MSE Mean: ", torch.mean(errors))




def get_args():
    parser = argparse.ArgumentParser(description='Train video network')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00005,
                        help='Learning rate', dest='lr')
    parser.add_argument('-a', '--audio_num', metavar='AN', type=int, nargs='?', default=1,
    help='Learning rate', dest='audio_num')
    parser.add_argument('-c', '--checkpoint-period', dest='chkpointperiod', type=int, default=5,
                    help='Number of epochs to save a checkpoint')
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

    train(lr=args.lr, device=device, chkpointperiod=args.chkpointperiod,audio_num = args.audio_num)