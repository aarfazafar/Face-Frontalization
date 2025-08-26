#data.py
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from random import shuffle

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png"])

def get_subdirs(directory):
    return sorted([os.path.join(directory, name) for name in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, name))])

class FaceDataset(Dataset):
    def __init__(self, imageset_dir, image_size=128, random_shuffle=False, max_samples=None):
        self.image_size = image_size
        self.frontals = [os.path.join(imageset_dir, f) for f in sorted(os.listdir(imageset_dir)) if is_image(f) and os.path.isfile(os.path.join(imageset_dir, f))]
        profile_files = [[os.path.join(person_dir, p) for p in sorted(os.listdir(person_dir)) if is_image(p) and os.path.isfile(os.path.join(person_dir, p))] for person_dir in get_subdirs(imageset_dir)]
        frontal_ind = []
        for ind, profiles in enumerate(profile_files):
            frontal_ind += [ind] * len(profiles)
        self.profiles = [item for sublist in profile_files for item in sublist]
        self.frontal_indices = frontal_ind
        if max_samples and max_samples < len(self.frontal_indices):
            self.frontal_indices = self.frontal_indices[:max_samples]
            self.profiles = self.profiles[:max_samples]
        if random_shuffle:
            ind = list(range(len(self.frontal_indices)))
            shuffle(ind)
            self.profiles = [self.profiles[i] for i in ind]
            self.frontal_indices = [self.frontal_indices[i] for i in ind]

    def __len__(self):
        return len(self.frontal_indices)

    def __getitem__(self, idx):
        profile = Image.open(self.profiles[idx]).convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        frontal = Image.open(self.frontals[self.frontal_indices[idx]]).convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        profile = np.array(profile).transpose((2, 0, 1)) / 127.5 - 1.0
        frontal = np.array(frontal).transpose((2, 0, 1)) / 127.5 - 1.0
        return profile, frontal

def get_data_loader(datapath, batch_size=10, max_samples=None, shuffle=True):
    dataset = FaceDataset(datapath, max_samples=max_samples, random_shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)