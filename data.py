import torch
from torch.utils.data import Dataset
import skvideo.io
import pudb
import numpy as np


class Video(Dataset):
    def __init__(self, video_num, num_items = 160000):
        self.num_items = num_items #num_items iis 160000 as specified in paper
        if video_num == 1:

            self.vid = torch.tensor(skvideo.io.vread("cat.mp4").astype(np.single) / 255.0)
        else:
            self.vid = "Define this later"
        
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
        coord_values = self.vid[rand_indices]
        return coord_values, vid_values
