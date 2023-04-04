import torch
from torch.utils.data import Dataset
import skvideo.io
from skvideo import datasets
import numpy as np
from scipy.io import wavfile
from PIL import Image
import scipy


class Video(Dataset):
    def __init__(self, video_num, num_items = 160000):
        self.num_items = num_items #num_items iis 160000 as specified in paper
        if video_num == 1:
            self.vid = torch.tensor(skvideo.io.vread("./data/cat_video.mp4").astype(np.single) / 255.0)
        else:
            self.vid = torch.tensor(skvideo.io.vread(datasets.bikes()).astype(np.single) / 255.0)

        self.num_frames = self.vid.shape[0]
        
        self.vid = (self.vid-0.5)*2 #scale video between -1 and 1

        self.original_shape = self.vid.shape #just keep this for later reconstruction
       

        #coordinates for each diemension
        coords_dim0 = torch.linspace(-1, 1, self.vid.shape[0])
        coords_dim1 = torch.linspace(-1, 1, self.vid.shape[1])
        coords_dim2 = torch.linspace(-1, 1, self.vid.shape[2])

        coords = torch.stack(torch.meshgrid((coords_dim0, coords_dim1, coords_dim2)), axis=-1)

        #pu.db
        self.coords = coords.view(-1,3)
        self.vid = self.vid.view(-1,3)

        print("Loaded data")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rand_indices = torch.randint(0, self.vid.shape[0], (self.num_items,)) 
        vid_values = self.vid[rand_indices]
        coord_values = self.coords[rand_indices]
        return coord_values, vid_values


class Audio(Dataset):
    def __init__(self, audio_num):
        if audio_num == 1:

            samplerate, self.audio = wavfile.read('./data/gt_bach.wav')
            self.audio = torch.unsqueeze(torch.tensor(self.audio), dim=1)
        else:
            samplerate, self.audio = wavfile.read('./data/gt_counting.wav')
            self.audio = torch.unsqueeze(torch.tensor(self.audio), dim=1)

        self.coords = torch.linspace(-1, 1, self.audio.shape[0]).unsqueeze(dim=1)*100 # the paper did this mulitiplication by 100


    def __len__(self):
        return 1

    def __getitem__(self, idx):
        audio_values = self.audio[:]
        coord_values = self.coords[:]
        return coord_values, audio_values

class Poisson(Dataset):
    def __init__(self, imageMult, num_items = None):
        self.num_items = num_items
        img = Image.open('./data/starfish.jpg').convert("L") #grayscale
        self.image = torch.tensor(np.array(img).astype(np.single))/255.0
       
        self.image = (self.image - 0.5)*2

        self.image *= imageMult

        self.original_shape = self.image.shape #just keep this for later reconstruction
        
        coords_dim0 = torch.linspace(-1, 1, self.image.shape[0])
        coords_dim1 = torch.linspace(-1, 1, self.image.shape[1])
        coords = torch.stack(torch.meshgrid((coords_dim0, coords_dim1)), axis=-1)
        self.coords = coords.view(-1,2)
        x_sobel = torch.tensor(scipy.ndimage.sobel(self.image, axis=0).astype(np.single))
        y_sobel = torch.tensor(scipy.ndimage.sobel(self.image, axis=1).astype(np.single))
        gradient = torch.stack((x_sobel, y_sobel), axis=2)


        self.laplace = torch.tensor(scipy.ndimage.laplace(self.image).astype(np.single))

        self.image = self.image.view(-1,1)
        self.gradient = gradient.view(-1,2)
        self.laplace = self.laplace.view(-1,1)




    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.num_items != 0:
            rand_indices = torch.randint(0, self.image.shape[0], (self.num_items,)) 
            coord_values = self.coords[rand_indices]
            image_values = self.image[rand_indices]
            gradient_values = self.gradient[rand_indices]
            laplace_values = self.laplace[rand_indices]
        else:
            coord_values = self.coords[:]
            image_values = self.image[:]
            gradient_values = self.gradient[:]
            laplace_values = self.laplace[:]
        return coord_values, image_values, gradient_values, laplace_values
