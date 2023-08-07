
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import tifffile as tiff
from PIL import Image 


class Cell_Dataset(Dataset):
    def __init__(self, cell_type, time, transforms=None):
        '''
        data - train data path
        masks - train masks path
        '''
        self.cell_type = cell_type
        self.t = time

        self.path = f'data/external/Cell Challenge/training_data/{self.cell_type}/{self.t}'
        
        self.transforms = transforms

        self.names = sorted(os.listdir(self.path))
        
    def __len__(self):
        
        return len(self.names)
        
    def __getitem__(self, idx):
        if idx > len(self.names):
            print("Index out of range for the dataset")
            return

        image_name = os.path.join(self.path, self.names[idx])
        mask_name = os.path.join(f'{self.path}_MK', self.names[idx])

        img = tiff.imread(image_name)
        mask = tiff.imread(mask_name)

        img = np.transpose(img, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))
        
        if self.transforms is not None:
            img = self.transforms(img)
            mask = self.transforms(mask)

        else:
            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask)
        
        img = img[0]
        mask = mask[0]

        img = torch.unsqueeze(img, dim=0)
        mask = torch.unsqueeze(mask, dim=0)

        return img, mask
    

#####################################################################


class Cell_Challenge_Segmentation_Dataset(Dataset):
    """
    Creates a Dataset out of all possible in http://celltrackingchallenge.net/ and prepares
    it to use it with a UNet Neural Network.

    Inputs:
        - cell_type: cell type of dataset to create.
        - transforms: transformation for data augmentation, preferably created with Albumentations
                      so that it can take as inputs both images and masks.
                      It is important that the images and masks are cropped in such a way that they
                      are compatible with the UNet.
        - test: boolean indicating wether its training or testing. If false, both images and masks
                will be taken in place.

    Outputs: a list containing [images, masks] properly transformed to tensors and cropped so that
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
        

    # def __init__(self, cell_type, time, transforms=None, test=False):

    #     self.cell_type = cell_type
    #     self.t = time

    #     if not test:
    #         self.image_path = f'data/external/Cell Challenge/training_data/{self.cell_type}/{self.t}'
    #         self.mask_path = f'data/external/Cell Challenge/training_data/{self.cell_type}/{self.t}_GT/SEG'
    #         self.names = sorted(os.listdir(self.mask_path))

    #     else:
    #         self.image_path = f'data/external/Cell Challenge/challenge_data/{self.cell_type}/{self.t}'
    #         self.names = sorted(os.listdir(self.image_path))
        
    #     self.transforms = transforms

    #     self.test = test
        
    # def __len__(self):
        
    #     return len(self.names)
        
    # def __getitem__(self, idx):

    #     if idx > len(self.names):
    #         print("Index out of range for the dataset")
    #         return
        
    #     if not self.test:
    #         name = self.names[idx].split(".")[0]
    #         split = name.split("_")
    #         image_idx = split[2]
    #         image_slice = split[3]

    #         image_name = os.path.join(self.image_path, f't{image_idx}.tif')
    #         img = tiff.imread(image_name)[int(image_slice)] # shape (443, 512)

    #         mask_name = os.path.join(self.mask_path, self.names[idx])
    #         mask = tiff.imread(mask_name).astype(np.float32) > 0
        
    #     else:
    #         slice = np.random.randint(0,5)
    #         image_name = os.path.join(self.image_path, self.names[idx])
    #         img = tiff.imread(image_name)[slice] # shape (443, 512)

    #     if self.transforms is not None:
    #         # Apply the augmentations to image and mask

    #         if not self.test:
    #             augmented = self.transforms(image=img, mask=mask)

    #             # Retrieve the augmented image and mask
    #             img = augmented['image']
    #             mask = augmented['mask']
    #             # img, mask = self.transforms([img, mask])
    #             # img, mask = torch.unsqueeze(img, 0), torch.unsqueeze(mask, 0)

    #             img = transforms.ToTensor()(img) # shape (1, 443, 512)
    #             mask = transforms.ToTensor()(mask) # shape (1, 443, 512)
                
    #             return img, mask
        
    #         else:
    #             img = transforms.ToTensor()(img)
    #             img = transforms.CenterCrop((432,512))(img)
                
    #             return img
    

#####################################################################


class Cell_Challenge_MaskRCNN_Dataset(Dataset):
    """
    Creates a Dataset out of all possible in http://celltrackingchallenge.net/ and prepares
    it to use it with Mask RCNN pretrained model from PyTorch.

    Inputs:
        - cell_type: cell type of dataset to create
        - transforms: transformation for data augmentation, preferably created with Albumentations
                      so that it can take as inputs both images and masks.
        - test: boolean indicating wether its training or testing. If false, both images and masks
                will be taken in place.
        - binary: boolean indicating wether perform semantic segmentation (if True) or instance.

    Outputs: a list containing [images, target] where the images are tensors with 3 channels (all of
             them are the same, but it is necessary for this NN) and targets contains:
             - boxes: coordinates for box detection of the instances
             - labels: vector of int values for different instances only if binary = False. If not,
                       the tensor will be of ones of length the number of instances.
             - masks: tensor of [number_instances, H, W] containing the mask for each of the instances.
                      If binary=True, number_instances=1 having a global mask.

    Remarks: Cell Challenge Datasets are divided in two folders (time 01 and 02) corresponding to
             different measures. This function merges both folders to create a unique dataset instead
             of creating two different ones and training the UNet in different steps.
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
    

# class Cell_Challenge_MaskRCNN_Dataset(Dataset):
#     """
#     Creates a Dataset out of all possible in http://celltrackingchallenge.net/ and prepares
#     it to use it with Mask RCNN pretrained model from PyTorch.

#     Inputs:
#         - cell_type: cell type of dataset to create
#         - transforms: transformation for data augmentation, preferably created with Albumentations
#                       so that it can take as inputs both images and masks.
#                       It is important that the images and masks are cropped in such a way that they
#                       are compatible with the UNet.
#         - test: boolean indicating wether its training or testing. If false, both images and masks
#                 will be taken in place.

#     Outputs: a list containing [images, masks] properly transformed to tensors and cropped so that
#              they can be feed to the UNet when in training and only images if in test mode.

#     Remarks: Cell Challenge Datasets are divided in two folders (time 01 and 02) corresponding to
#              different measures. This function merges both folders to create a unique dataset instead
#              of creating two different ones and training the UNet in different steps.
#     """

#     def __init__(self, cell_type, time, transforms=None, test=False, binary=True):

#         self.cell_type = cell_type
#         self.t = time

#         if not test:
#             self.image_path = f'data/external/Cell Challenge/training_data/{self.cell_type}/{self.t}'
#             self.mask_path = f'data/external/Cell Challenge/training_data/{self.cell_type}/{self.t}_GT/SEG'
#             self.names = sorted(os.listdir(self.mask_path))
        
#         else:
#             self.image_path = f'data/external/Cell Challenge/challenge_data/{self.cell_type}/{self.t}'
#             self.names = sorted(os.listdir(self.image_path))

#         self.binary = binary

#         self.transforms = transforms

#         self.test = test
        
#     def __len__(self):
        
#         return len(self.names)
        
#     def __getitem__(self, idx):

#         if idx > len(self.names):
#             print("Index out of range for the dataset")
#             return
        
#         if self.test:
#             slice = np.random.randint(0,5)
#             image_name = os.path.join(self.image_path, self.names[idx])
#             img = tiff.imread(image_name)[slice] # shape (443, 512)
#             img = transforms.ToTensor()(img) # shape (1, 443, 512)

#             return img
        
#         name = self.names[idx].split(".")[0]
#         split = name.split("_")
#         image_idx = split[2]
#         image_slice = split[3]

#         image_name = os.path.join(self.image_path, f't{image_idx}.tif')
#         img = tiff.imread(image_name)[int(image_slice)] # shape (443, 512)

#         mask_name = os.path.join(self.mask_path, self.names[idx])
#         mask = Image.open(mask_name)
#         mask = np.array(mask) # shape (443, 512)
#         # mask = tiff.imread(mask_name).astype(np.float32) > 0

#         obj_ids = np.unique(mask)
#         obj_ids = obj_ids[1:]
#         num_objs = len(obj_ids)

#         if self.binary:
#             mask = mask > 0

#         if self.transforms is not None:

#             num_objs = 0
#             while num_objs < 1:
#                 # Apply the augmentations to image and mask
#                 augmented = self.transforms(image=img, mask=mask)

#                 # Retrieve the augmented image and mask
#                 img_augmented = augmented['image']
#                 mask_augmented = augmented['mask']

#                 obj_ids = np.unique(mask_augmented)
#                 obj_ids = obj_ids[1:]
#                 num_objs = len(obj_ids)
        
#             img = img_augmented
#             mask = mask_augmented

#         masks = np.zeros((num_objs , mask.shape[0] , mask.shape[1]))
#         boxes = []
#         for i,obj in enumerate(obj_ids):
#             masks[i][mask == obj] = True

#             pos = np.where(masks[i])
#             xmin = np.min(pos[1])
#             xmax = np.max(pos[1])
#             ymin = np.min(pos[0])
#             ymax = np.max(pos[0])
#             boxes.append([xmin , ymin , xmax , ymax])
            
#         boxes = torch.as_tensor(boxes , dtype = torch.float32)
#         labels = torch.ones((num_objs,) , dtype = torch.int64)
        
#         img = Image.fromarray(img).convert("RGB") # shape: H and W inversed
#         # img = torch.as_tensor(img, dtype=torch.float32)
#         # img = torch.unsqueeze(img, dim=0)
#         mask = torch.as_tensor(masks, dtype=torch.uint8)
        
#         target = {}
#         target["boxes"] = boxes
#         target["labels"] = labels
#         target["masks"] = mask

#         return transforms.ToTensor()(img), target