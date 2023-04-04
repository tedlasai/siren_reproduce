1. Model is provided in the siren.py and meta_siren.py(used for reconstruction experiment) file
2. Train/test scripts for each of the 4 experiments are provided.
3. Additionally there are some dataloader files and utilities files.
4. I have included data files in the git repo for the first 3 tasks, but for the fourth task you will need to download CelebA. 
5. dataio.py contains the dataloader I copied from the original SIREN authors.

As mentioned in the write up this git repo targets 4 tasks. We provide train/test scripts for all these tasks.
1. Audio representation
2. Video representation
3. Poisson reconstruction
4. Learning an implicit space(reconstruction/impainting)

>📋  Code accompanying my reconstruction of SIREN.

# Reimplementation of SIREN

## Requirements

To build environemnt:

```setup
conda env create -f env.yml
```


## Training

To train the model(s) in the paper, run command such as:

```train
python train/train_audio.py
python train/train_video.py
```

>There are 4 different training scripts. One trianing script for each experiment. 

## Evaluation

To evaluate my model on a sample task such as audio:

```eval
python test/test_audio.py -c checkpoint_path.pth
```

> There are 4 test scripts. One for each experiment.




