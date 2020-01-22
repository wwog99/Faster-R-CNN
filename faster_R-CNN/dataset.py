from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

"""
Absolute paths
"""
image_path = "processed/processed_img_"
label_path = 'processed/processed_label.npy'


"""
DataLoader of image and labels
"""
class Image_loader(Dataset):
    def __init__(self):
        self.len = 6
        self.image_set = torch.zeros([6, 3, 512, 512], dtype=torch.int32)
        for i in range(6):
            image = np.load(image_path + str(i + 1) + ".npy") #image file form [512,512,3]
            image = image.transpose([2,0,1]) # [512,512,3] to [3,512,512](for making form of [N,C,W,H]
            image = torch.Tensor(image)
            self.image_set[i] = image
        self.label = np.load(label_path, allow_pickle= True)

    def __getitem__(self, index):
        return self.image_set[index].reshape([1,3,512,512]), self.label[index]

    def __len__(self):
        return self.len
