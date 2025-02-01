import torch
import numpy as np
from torch.utils.data import Dataset

class SLdataset(Dataset):
    def __init__(self, image_file, mask_file, transform=None):
        self.image_data = np.load(image_file)  # Load the images from the .npy file
        self.mask_data = np.load(mask_file)    # Load the masks from the .npy file
        self.transform = transform

    def __len__(self):
        return self.image_data.data.shape[0]
    

    def __getitem__(self, index):
        #image = self.image_data[index]  # Load the image at the current index
        image = np.transpose(self.image_data[index], (2, 0, 1))
        #mask = self.mask_data[index]    # Load the mask at the current index
        mask = np.transpose(self.mask_data[index], (2, 0, 1))

        return image, mask


# if __name__ == "__main__": 
#     #is used in Python scripts to ensure that certain parts of the script only run when the script is executed directly
#     dataset = SLdataset("trainimages.npy", "trainmasks.npy")
#     print("Image data shape:", dataset.image_data.shape)
#     print("Mask data shape:", dataset.mask_data.shape)
#     print(dataset.__len__())

#     index = 0
#     image, mask = dataset[index]

#     # Print the shapes directly
#     print(f"Image shape at index {index}: {image.shape}")
#     print(f"Mask shape at index {index}: {mask.shape}")