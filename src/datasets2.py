import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import tifffile as tiff
from PIL import Image
import cv2

class Embed_Segmentation_Dataset(Dataset):
    """
    Creates a Dataset out of all possible in http://celltrackingchallenge.net/ and prepares
    it to use it with a UNet Neural Network.

    Inputs:
        - cell_type (str): cell type of dataset to create.
        - transforms (albumentations.Compose): transformation for data augmentation taking as inputs
                both images and masks. It is important that the images and masks are cropped in such a
                way that they are compatible with the UNet arquitecture.
        - test (bool): indicating wether its training or testing. If false, both images and masks
                will be taken in place.

    Outputs (list): containing [images, masks] properly transformed to tensors and cropped so that
             they can be feed to the UNet when in training and only images if in test mode.

    Remarks: Cell Challenge Datasets are divided in two folders (time 01 and 02) corresponding to
             different measures. This function merges both folders to create a unique dataset instead
             of creating two different ones and training the UNet in different steps.
    """

    def __init__(self, transforms=None, test=False):

        if not test:
            self.image_path = f'data/external/EmbedSeg/Mouse-Skull-Nuclei-CBG/train/images'
            self.mask_path = f'data/external/EmbedSeg/Mouse-Skull-Nuclei-CBG/train/masks'

        else:
            self.image_path = f'data/external/EmbedSeg/Mouse-Skull-Nuclei-CBG/test/images'
            self.mask_path = f'data/external/EmbedSeg/Mouse-Skull-Nuclei-CBG/test/masks'

        self.image_names = sorted(os.listdir(self.image_path))
        self.masks_names = sorted(os.listdir(self.mask_path))
        self.transforms = transforms

        self.test = test
        
    def __len__(self):
        
        return len(self.image_names)
        
    def __getitem__(self, idx):

        # Transformation for test mode or self.transform = None
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        image_name = os.path.join(self.image_path, self.image_names[idx])
        img = tiff.imread(image_name).astype(np.float16)         # shape (z, 443, 512)
        slice = np.random.randint(0,img.shape[0])
        img = transform(img[slice])                  # shape (1, 443, 512)

        # Training mode
        mask_name = os.path.join(self.mask_path, self.masks_names[idx])
        mask = tiff.imread(mask_name)[slice].astype(np.float32) > 0  # shape (443, 512)
        mask = transform(mask)  

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)

            # Retrieve the augmented image and mask with shape depending on Crop of transformation
            img = augmented['image']  # shape (H, W)
            mask = augmented['mask']  # shape (H, W)

            # This dataset contains only one channel intead of 3
            img = transforms.ToTensor()(img) # shape (1, H, W)
            mask = transforms.ToTensor()(mask) # shape (1, H, W)
            
            return img, mask
    
        else:

            return img, mask
    

#####################################################################


class Bowl_Kaggle_Segmentation_Dataset(Dataset):
    """
    Creates a Dataset from the 2018 Data Science Bowl available in
    https://www.kaggle.com/c/data-science-bowl-2018/data and prepares
    it to use it with a UNet Neural Network.

    Inputs:
        - transforms (albumentations.Compose): transformation for data augmentation taking as inputs
                both images and masks. It is important that the images and masks are cropped in such a
                way that they are compatible with the UNet arquitecture.
        - type (str): indicating wether its training or testing. If 'train', both images and masks
                will be taken in place.

    Outputs (list): containing [images, masks] properly transformed to tensors and cropped so that
             they can be feed to the UNet when in training and only images if in test mode.

    Remarks:
    """

    def __init__(self, transforms=None, type='train'):

        images_path = f'data/external/2D/Science Bowl Kaggle/{type}/images'
        masks_path = f'data/external/2D/Science Bowl Kaggle/{type}/masks'
        images_total = sorted(os.listdir(images_path))
        masks_total = sorted(os.listdir(masks_path))[:670]

        images_names = []
        masks_names = []

        for i, names in enumerate(images_total):
            img = cv2.imread(os.path.join(images_path,names), cv2.IMREAD_GRAYSCALE)

            if img.shape == (256,256) or img.shape == (256,320):
                img_path = os.path.join(images_path,names)
                images_names.append(img_path)

                if type == 'train':
                    mask_path = os.path.join(masks_path,masks_total[i])
                    masks_names.append(mask_path)
        
        self.images_names = images_names
        self.masks_names = masks_names

        means = [np.mean(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in images_names]
        stds = [np.std(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in images_names]
        
        means = np.array(means)
        stds = np.array(stds)

        self.mean = np.mean(means)
        self.std = np.mean(stds)

        self.transforms = transforms

        self.test = type == 'test'
        
    def __len__(self):
        
        return len(self.images_names)
        
    def __getitem__(self, idx):

        # img = Image.open(os.path.join(self.images_path,self.images_names[idx]))
        # img = np.array(img)[:,:,0]
        # img = cv2.imread(os.path.join(self.images_path,self.images_names[idx]))
        img = cv2.imread(self.images_names[idx], cv2.IMREAD_GRAYSCALE)
        # img = (img - self.mean) / self.std

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256))
        ])

        if self.test:

            return transform(img)

        # Training mode
        # mask = Image.open(os.path.join(self.masks_path,self.masks_names[idx]))
        # mask = np.array(mask)
        # mask = cv2.imread(os.path.join(self.masks_path,self.masks_names[idx]))
        mask = cv2.imread(self.masks_names[idx], cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)

            # Retrieve the augmented image and mask with shape depending on Crop of transformation
            img = augmented['image']  # shape (H, W)
            mask = augmented['mask']  # shape (H, W)

            # This dataset contains only one channel intead of 3
            img = transforms.ToTensor()(img) # shape (1, H, W)
            mask = transforms.ToTensor()(mask) # shape (1, H, W)
            
            return img, mask
    
        else:
            img = transform(img)
            mask = transform(mask)

            return img, mask
    

#####################################################################


class WarwickQU_Segmentation_Dataset(Dataset):
    """
    Creates a Dataset from the 2015 MICCAI challenge available in
    https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download and prepares
    it to use it with a UNet Neural Network.Bowl_KaggleBowl_Kaggle

    Inputs:
        - transforms (albumentations.Compose): transformation for data augmentation taking as inputs
                both images and masks. It is important that the images and masks are cropped in such a
                way that they are compatible with the UNet arquitecture.
        - type (str): indicating wether its training or testing. If 'train', both images and masks
                will be taken in place.

    Outputs (list): containing [images, masks] properly transformed to tensors and cropped so that
             they can be feed to the UNet when in training and only images if in test mode.

    Remarks:
    """

    def __init__(self, size=(512,768), channels=3, normalize=True, transforms=None, type='train'):

        images_path = f'data/external/2D/Warwick QU/{type}/images'
        masks_path = f'data/external/2D/Warwick QU/{type}/masks'
        images_total = sorted(os.listdir(images_path))
        masks_total = sorted(os.listdir(masks_path))

        images_names = []
        masks_names = []

        if channels == 3:
            flags = cv2.IMREAD_COLOR

        elif channels == 1:
            flags = cv2.IMREAD_GRAYSCALE
        
        else:
            print("Incorrect number of channels")
            return
    
        self.flags = flags

        for i, names in enumerate(images_total):
            img = cv2.imread(os.path.join(images_path,names), flags)

            if img.shape[:2] == (522,775):
                img_path = os.path.join(images_path,names)
                images_names.append(img_path)

                if type == 'train':
                    mask_path = os.path.join(masks_path,masks_total[i])
                    masks_names.append(mask_path)
        
        self.images_names = images_names
        self.masks_names = masks_names

        if normalize:
            means = [np.mean(cv2.imread(name, flags), axis=(0,1)) for name in images_names]
            stds = [np.std(cv2.imread(name, flags), axis=(0,1)) for name in images_names]
            
            means = np.array(means)
            stds = np.array(stds)

            self.mean = np.mean(means, axis=0)
            self.std = np.mean(stds, axis=0)
        
        self.size = size
        self.normalize = normalize
        self.transforms = transforms
        self.test = type == 'test'
        
    def __len__(self):
        
        return len(self.images_names)
        
    def __getitem__(self, idx):

        img = cv2.imread(self.images_names[idx], self.flags)

        if self.normalize:
            img = (img - self.mean) / self.std

        # Transformation for test mode or self.transform = None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.size)
        ])

        if self.test:

            return transform(img)

        # Training mode
        mask = cv2.imread(self.masks_names[idx], cv2.IMREAD_GRAYSCALE)
        
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)

            # Retrieve the augmented image and mask with shape depending on Crop of transformation
            img = augmented['image']  # shape (H, W)
            mask = augmented['mask']  # shape (H, W)

            # This dataset contains only one channel intead of 3
            img = transforms.ToTensor()(img) # shape (1, H, W)
            mask = transforms.ToTensor()(mask) # shape (1, H, W)
            
            return img, mask
    
        else:
            img = transform(img)
            mask = transform(mask)

            return img, mask
    

#####################################################################


class Cell_Challenge_Segmentation_Dataset(Dataset):
    """
    Creates a Dataset out of all possible in http://celltrackingchallenge.net/ and prepares
    it to use it with a UNet Neural Network.

    Inputs:
        - cell_type (str): cell type of dataset to create.
        - transforms (albumentations.Compose): transformation for data augmentation taking as inputs
                both images and masks. It is important that the images and masks are cropped in such a
                way that they are compatible with the UNet arquitecture.
        - test (bool): indicating wether its training or testing. If false, both images and masks
                will be taken in place.

    Outputs (list): containing [images, masks] properly transformed to tensors and cropped so that
             they can be feed to the UNet when in training and only images if in test mode.

    Remarks: Cell Challenge Datasets are divided in two folders (time 01 and 02) corresponding to
             different measures. This function merges both folders to create a unique dataset instead
             of creating two different ones and training the UNet in different steps.
    """

    def __init__(self, cell_type, transforms=None, test=False):

        self.cell_type = cell_type

        if not test:
            self.image_path1 = f'data/external/Cell Challenge/training_data/{self.cell_type}/01'
            self.image_path2 = f'data/external/Cell Challenge/training_data/{self.cell_type}/02'
            self.mask_path1 = f'data/external/Cell Challenge/training_data/{self.cell_type}/01_ST/SEG'
            self.mask_path2 = f'data/external/Cell Challenge/training_data/{self.cell_type}/02_ST/SEG'
            self.names1 = sorted(os.listdir(self.mask_path1))
            self.names2 = sorted(os.listdir(self.mask_path2))

        else:
            self.image_path1 = f'data/external/Cell Challenge/test_data/{self.cell_type}/01'
            self.image_path2 = f'data/external/Cell Challenge/test_data/{self.cell_type}/02'
            self.names1 = sorted(os.listdir(self.image_path1))
            self.names2 = sorted(os.listdir(self.image_path2))
        
        self.transforms = transforms

        self.test = test
        
    def __len__(self):
        
        return len(self.names1 + self.names2)
        
    def __getitem__(self, idx):

        # Transformation for test mode or self.transform = None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((432,512))
        ])

        if self.test:
            # slice = np.random.randint(0,5)  # selects a random slice in Z
            slice = 0

            if idx < len(self.names1):
                image_name = os.path.join(self.image_path1, self.names1[idx])
            
            else:
                image_name = os.path.join(self.image_path2, self.names2[idx-len(self.names1)])
                
            img = tiff.imread(image_name)[slice]  # shape (443, 512)
            img = transform(img)                  # shape (1, 443, 512)

            return img

        # Training mode
        if idx < len(self.names1):
            name = self.names1[idx].split(".")[0]  # [man_segidx, tif]

        else:
            name = self.names2[idx-len(self.names1)].split(".")[0]

        # Extract image index and slice out of mask file name
        split = name.split("_")   # [man, segidx]
        image_idx = split[1][3:]  # idx
        # image_slice = split[3]   # slice

        # Path to images and mask corresponding to the same measure
        if idx < len(self.names1):
            image_name = os.path.join(self.image_path1, f't{image_idx}.tif')
            mask_name = os.path.join(self.mask_path1, self.names1[idx])
        
        else:
            image_name = os.path.join(self.image_path2, f't{image_idx}.tif')
            mask_name = os.path.join(self.mask_path2, self.names2[idx-len(self.names1)])

        img = tiff.imread(image_name)[0]                         # shape (443, 512)
        mask = tiff.imread(mask_name)[0].astype(np.float32) > 0  # shape (443, 512)
        
        if self.transforms is not None:
            augmented = self.transforms(image=img, mask=mask)

            # Retrieve the augmented image and mask with shape depending on Crop of transformation
            img = augmented['image']  # shape (H, W)
            mask = augmented['mask']  # shape (H, W)

            # This dataset contains only one channel intead of 3
            img = transforms.ToTensor()(img) # shape (1, H, W)
            mask = transforms.ToTensor()(mask) # shape (1, H, W)
            
            return img, mask
    
        else:
            img = transform(img)
            mask = transform(mask)

            return img, mask
    

#####################################################################


class Cell_Challenge_MaskRCNN_Dataset(Dataset):
    """
    Creates a Dataset out of all possible in http://celltrackingchallenge.net/ and prepares
    it to use it with Mask RCNN pretrained model from PyTorch.

    Inputs:
        - cell_type (str): cell type of dataset to create.
        - transforms (albumentations.Compose): transformation for data augmentation taking as inputs
                both images and masks.
        - test (bool): indicating wether its training or testing. If false, both images and masks
                will be taken in place.
        - binary (bool): indicating wether perform semantic (true) or instance segmentation.

    Outputs (list): containing [images, target] where the images are tensors with 3 channels and targets
        contains:
             - boxes: coordinates for box detection of the instances.
             - labels: vector of int values for different instances only if binary = False. If not,
                       the tensor will be of ones of length the number of instances.
             - masks: tensor of [number_instances, H, W] containing the mask for each of the instances.
                      If binary=True, number_instances=1 having a global mask.

    Remarks: Cell Challenge Datasets are divided in two folders (time 01 and 02) corresponding to
             different measures. This function merges both folders to create a unique dataset instead
             of creating two different ones and training the UNet in different steps.
             I this case, the 3 channels of the images are the same (since the original images are in
             black and white) but doing so is neccesary to implement the Mask RCNN arquitecture.
    """

    def __init__(self, cell_type, transforms=None, test=False, binary=True):

        self.cell_type = cell_type

        if not test:
            self.image_path1 = f'data/external/Cell Challenge/training_data/{self.cell_type}/01'
            self.image_path2 = f'data/external/Cell Challenge/training_data/{self.cell_type}/02'
            self.mask_path1 = f'data/external/Cell Challenge/training_data/{self.cell_type}/01_GT/SEG'
            self.mask_path2 = f'data/external/Cell Challenge/training_data/{self.cell_type}/02_GT/SEG'
            self.names1 = sorted(os.listdir(self.mask_path1))
            self.names2 = sorted(os.listdir(self.mask_path2))

        else:
            self.image_path1 = f'data/external/Cell Challenge/test_data/{self.cell_type}/01'
            self.image_path2 = f'data/external/Cell Challenge/test_data/{self.cell_type}/02'
            self.names1 = sorted(os.listdir(self.image_path1))
            self.names2 = sorted(os.listdir(self.image_path2))

        self.binary = binary

        self.transforms = transforms

        self.test = test
        
    def __len__(self):
        
        return len(self.names1 + self.names2)
        
    def __getitem__(self, idx):

        if self.test:
            # slice = np.random.randint(0,5)  # selects a random slice in Z
            slice = 0

            if idx < len(self.names1):
                image_name = os.path.join(self.image_path1, self.names1[idx])
            
            else:
                image_name = os.path.join(self.image_path2, self.names2[idx-len(self.names1)])
                
            img = tiff.imread(image_name)[slice]  # shape (443, 512)
            img = transforms.ToTensor()(img) # shape (1, 443, 512)

            return img
        
        # Training mode
        if idx < len(self.names1):
            name = self.names1[idx].split(".")[0]  # [man_seg_idx_slice, tif]

        else:
            name = self.names2[idx-len(self.names1)].split(".")[0]
        
        # Extract image index and slice out of mask file name
        split = name.split("_")  # [man_seg, idx, slice]
        image_idx = split[2]     # idx
        image_slice = split[3]   # slice

        # Path to images and mask corresponding to the same measure
        if idx < len(self.names1):
            image_name = os.path.join(self.image_path1, f't{image_idx}.tif')
            mask_name = os.path.join(self.mask_path1, self.names1[idx])
        
        else:
            image_name = os.path.join(self.image_path2, f't{image_idx}.tif')
            mask_name = os.path.join(self.mask_path2, self.names2[idx-len(self.names1)])

        img = tiff.imread(image_name)[int(image_slice)]   # shape (443, 512)
        mask = tiff.imread(mask_name).astype(np.float32)  # shape (443, 512)

        obj_ids = np.unique(mask)  # instances in the mask 
        obj_ids = obj_ids[1:]      # background doesn't count
        num_objs = len(obj_ids)    # total number of instances

        # Only semantic (not instance) segmentation
        if self.binary:
            mask = mask > 0

        if self.transforms is not None:
            num_objs = 0

            # Perform data augmentation assuring that there is a cell in the image
            while num_objs < 1:
                augmented = self.transforms(image=img, mask=mask)

                # Retrieve the augmented image and mask
                img_augmented = augmented['image']
                mask_augmented = augmented['mask']

                # Check if there are instances in the augmentations
                obj_ids = np.unique(mask_augmented)
                obj_ids = obj_ids[1:]
                num_objs = len(obj_ids)
        
            # H and W correspond to the Crop made. If not, (443,512)
            img = img_augmented    # shape (H, W)
            mask = mask_augmented  # shape (H, W)

        # Tensor with mask for each of the instances
        masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1]))
        boxes = []
        for i,obj in enumerate(obj_ids):
            masks[i][mask == obj] = True

            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin , ymin , xmax , ymax])
            
        boxes = torch.as_tensor(boxes , dtype = torch.float32)
        labels = torch.ones((num_objs,) , dtype = torch.int64)
        
        img = Image.fromarray(img).convert("RGB")         # shape (3, W, H)
        img = transforms.ToTensor()(img)                  # shape (3, H, W)
        mask = torch.as_tensor(masks, dtype=torch.uint8)  # shape (1, H, W)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = mask

        return img, target