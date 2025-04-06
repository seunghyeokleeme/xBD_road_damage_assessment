import os
from PIL import Image
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset

class xbdDataset(VisionDataset):
    def __init__(self, root = None, transforms = None, transform = None, target_transform = None):
        super(xbdDataset, self).__init__(root, transforms, transform, target_transform)
        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "targets")
        self.pre_image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith('pre_disaster.png')])
        self.post_image_filenames = sorted([f for f in os.listdir(self.images_dir) if f.endswith('post_disaster.png')])
        self.mask_filenames = sorted([f for f in os.listdir(self.masks_dir) if f.endswith('post_disaster_target.png')])
        
        assert len(self.pre_image_filenames) == len(self.post_image_filenames)
        assert len(self.post_image_filenames) == len(self.mask_filenames)

    def __len__(self):
        return len(self.mask_filenames)
    
    def __getitem__(self, index):
        pre_image_path = os.path.join(self.images_dir, self.pre_image_filenames[index])
        post_image_path = os.path.join(self.images_dir, self.post_image_filenames[index])
        mask_path = os.path.join(self.masks_dir, self.mask_filenames[index])

        pre_image = Image.open(pre_image_path).convert("RGB")
        post_image = Image.open(post_image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # [0, 1, 2], 0: background, 1: road, 2: damaged road
            
        if self.transform is not None:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
            if len(mask.shape) == 3:
                mask = mask.squeeze(0)

        # Calculate difference after converting to tensors
        diff_image = post_image - pre_image

        return diff_image, mask
