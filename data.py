import torch
from torch.utils.data import Dataset
import skvideo.io
from skvideo import datasets
import numpy as np
from scipy.io import wavfile



class Video(Dataset):
    def __init__(self, video_num, num_items = 160000):
        self.num_items = num_items #num_items iis 160000 as specified in paper
        if video_num == 1:
            self.vid = torch.tensor(skvideo.io.vread("cat_video.mp4").astype(np.single) / 255.0)
        else:
            self.vid = torch.tensor(skvideo.io.vread(datasets.bikes()).astype(np.single) / 255.0)

        self.num_frames = self.vid.shape[0]
        
        self.vid = (self.vid-0.5)*2 #scale video between -1 and 1
       

        #coordinates for each diemension
        coords_dim0 = torch.linspace(-1, 1, self.vid.shape[0])
        coords_dim1 = torch.linspace(-1, 1, self.vid.shape[1])
        coords_dim2 = torch.linspace(-1, 1, self.vid.shape[2])

        coords = torch.stack(torch.meshgrid((coords_dim0, coords_dim1, coords_dim2)), axis=-1)


        self.coords = coords.view(-1,3)
        self.vid = self.vid.view(-1,3)

        print("Loaded data")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        rand_indices = torch.randint(0, self.vid.shape[0], (self.num_items,)) 
        vid_values = self.vid[rand_indices]
        coord_values = self.coords[rand_indices]
        #pu.db
        return coord_values, vid_values


class Audio(Dataset):
    def __init__(self, audio_num):
        if audio_num == 1:

            samplerate, self.audio = wavfile.read('./gt_bach.wav')
            self.audio = torch.unsqueeze(torch.tensor(self.audio), dim=1)
        else:
            samplerate, self.audio = wavfile.read('./gt_counting.wav')
            self.audio = torch.unsqueeze(torch.tensor(self.audio), dim=1)
        
        #self.audio = (self.audio-0.5)*2 #scale video between -1 and 1
       

        #coordinates for each diemension
        self.coords = torch.linspace(-1, 1, self.audio.shape[0]).unsqueeze(dim=1)*100 # the paper did this mulitiplication by 100

        #self.vid = self.vid.view(-1,3)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        audio_values = self.audio[:]
        coord_values = self.coords[:]
        return coord_values, audio_values
