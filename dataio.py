import csv
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import torch

class CelebA(Dataset):
    def __init__(self, split, downsampled=True):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val'], "Unknown split"

        self.root = './celeba_dataset/img_align_celeba/img_align_celeba'
        self.img_channels = 3
        self.fnames = []

        with open('./celeba_dataset/list_eval_partition.csv', newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rowreader:
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0])

        self.downsampled = downsampled
        self.test_sparsity = "random"
        self.generalization_mode = "train"
        self.train_sparsity_range = (10,200)
        self.sidelength = (32, 32)

    def __len__(self):
        return len(self.fnames)
    

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions

            s = min(width, height)
            left = (width - s) / 2
            top = (height - s) / 2
            right = (width + s) / 2
            bottom = (height + s) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((32, 32))

        spatial_img = torch.tensor(np.array(img).astype(np.single))
        spatial_img = torch.moveaxis(spatial_img, (0, 1, 2), (1,2,0))
        
        #coordinates for each diemension
        coords_dim0 = torch.linspace(-1, 1, spatial_img.shape[0])
        coords_dim1 = torch.linspace(-1, 1, spatial_img.shape[1])

        coords = torch.stack(torch.meshgrid((coords_dim0, coords_dim1)), axis=-1)



        
       

        if self.test_sparsity == 'full':
                img_sparse = spatial_img
        elif self.test_sparsity == 'half':
            img_sparse = spatial_img
            img_sparse[:, 16:, :] = 0.
        else:
            if self.generalization_mode == 'conv_cnp_test':
                num_context = int(self.test_sparsity)
            else:
                num_context = int(
                    torch.empty(1).uniform_(self.train_sparsity_range[0], self.train_sparsity_range[1]).item())
            mask = spatial_img.new_empty(
                1, spatial_img.size(1), spatial_img.size(2)).bernoulli_(p=num_context / np.prod(self.sidelength))
            img_sparse = mask * spatial_img

        return coords, img_sparse, spatial_img

        
        
       