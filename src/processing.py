import os
import numpy as np
import cv2
from PIL import Image


def normalize_images(name, channels, start=1):
    """
    Creates a new folder called 'normalized' inside test and train folders for the dataset desired.
    The folder contains the numpy arrays (.npy) associated to the N(0,1) normalization of the images
    done channel wise (it takes care wether the input images have 1 or 3 channels).

    Inputs:
        - name (str): name of the dataset and folder one wants to normalize (i.e. 'Warwick QU/train')
        - channels (int): indicating the number of channels of the images associated to 'name/images'.
        - start (int=1): indicating in what number to start naming the files of the new folder.

    Outputs:

    Remarks: the 'start' value is not needed and makes no difference in the normalization, but it helps
        for associating images/normalization and masks. Some datasets start counting in 0 and others in 1.
    """

    path = f'data/external/2D/{name}'

    norm_path = os.path.join(path, 'normalized')
    if os.path.isdir(norm_path) and len(os.listdir(norm_path)) != 0:
        print(f"This dataset has already been normalized")
        return
    
    else:
        if not os.path.isdir(norm_path):
            os.makedirs(norm_path)

    names = os.listdir(os.path.join(path, 'images'))
    imgs_path = os.path.join(path, 'images')
    paths = [os.path.join(imgs_path, img_path) for img_path in names]

    if channels == 3:
        channels = cv2.IMREAD_COLOR

    elif channels == 1:
        channels = cv2.IMREAD_GRAYSCALE
    
    else:
        print("Incorrect number of channels")
        return
    
    means = [np.mean(cv2.imread(path, channels), axis=(0,1)) for path in paths]
    stds = [np.std(cv2.imread(path, channels), axis=(0,1)) for path in paths]
    
    means = np.array(means)
    stds = np.array(stds)

    mean = np.mean(means, axis=0)
    std = np.mean(stds, axis=0)

    for i, path in enumerate(paths, start=start):
        img = cv2.imread(path, channels)
        img = (img - mean) / std
        np.save(f"{norm_path}/norm_{i}.npy", img)

    return


#####################################################################


def processWarwick(path):
    """
    Organizes the images and masks of the 2015 MICCAI Warwick dataset available in
    https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/download in two different folders
    containing the images and masks separated. It moves the files to new folder instead of
    copying them.

    Inputs:
        - path (str): containing the local path to the dataset.

    Outputs:

    Remarks:
    """

    src = ['train', 'test']
    dst = ['images', 'masks']

    for folder in src:
        files = os.listdir(os.path.join(path,folder))

        for subfolder in dst:
            if not os.path.isdir(os.path.join(path,folder,subfolder)):
                os.makedirs(os.path.join(path,folder,subfolder))
            
        for file in files:
            if 'anno' in file:
                cmd = f'cp "{os.path.join(path,folder,file)}" "{os.path.join(path,folder,"masks")}"'
            
            else:
                cmd = f'cp "{os.path.join(path,folder,file)}" "{os.path.join(path,folder,"images")}"'

            rm = f'rm "{os.path.join(path,folder,file)}"'
            os.system(cmd)
            os.system(rm)

    return


#####################################################################


def processChallenge(path, type='train'):
    """
    Organizes the images and masks of a specific dataset from the Cell Tracking Challenge
    available in http://celltrackingchallenge.net/ in two different folders containing the
    images and masks separated. It moves the files to new folder instead of copying them.

    Inputs:
        - path (str): containing the local path to the dataset.

    Outputs:

    Remarks:
    """

    dsts = ['images', 'masks']
    folders = ['01', '02']

    if type == 'test':
        dsts = ['images']

    for dst in dsts:
        if not os.path.isdir(os.path.join(path,dst)):
            os.makedirs(os.path.join(path,dst))
        
        if dst == 'masks':
            folders = ['01_ST/SEG', '02_ST/SEG']

        i = 0
        for folder in folders:

            names = os.listdir(os.path.join(path, folder))

            for file in names:
                cmd = f'cp "{os.path.join(path, folder, file)}" "{os.path.join(path, dst, f"t{i}.tif")}"'
                os.system(cmd)
                i += 1


#####################################################################


def processBowl(path):
    """
    Organizes the images and masks from the Bowl Science Kaggl dataset available in
    https://www.kaggle.com/c/data-science-bowl-2018/data in two different folders containing
    the images and masks separated. It moves the files to new folder instead of copying them.

    Inputs:
        - path (str): containing the local path to the dataset.

    Outputs:

    Remarks:
    """

    src = ['train', 'test']
    dst = ['images', 'masks']

    for folder in src:
        files = os.listdir(os.path.join(path,folder))

        for subfolder in dst:
            if not os.path.isdir(os.path.join(path,folder,subfolder)):
                os.makedirs(os.path.join(path,folder,subfolder))
            
        for i,file in enumerate(files):
            if folder == 'train':
                subpath = os.path.join(path,folder,file,'images')
                name = os.listdir(subpath)
                img = os.path.join(path,folder,file,'images',name[0])

                copy = f'cp "{img}" "{os.path.join(path,folder,"images",f"{folder}_{i}.png")}"'
                delete = f'rm "{img}"'

                os.system(copy)
                os.system(delete)
                
                subpath = os.path.join(path,folder,file,'masks')
                names = os.listdir(subpath)

                for j,name in enumerate(names):
                    mask = os.path.join(path,folder,file,'masks',name)

                    copy = f'cp "{mask}" "{os.path.join(path,folder,"masks",f"{folder}_{i}_{j}.png")}"'
                    delete = f'rm "{mask}"'

                    os.system(copy)
                    os.system(delete)
            
            else:
                subpath = os.path.join(path,folder,file,'images')
                name = os.listdir(subpath)
                img = os.path.join(subpath,name[0])

                copy = f'cp "{img}" "{os.path.join(path,folder,"images",f"{folder}_{i}.png")}"'
                delete = f'rm "{img}"'

                os.system(copy)
                os.system(delete)

            filepath = os.path.join(path,folder,file)  
            delete = f'rm -r "{filepath}"'
            os.system(delete)

    return


#####################################################################


def maskBowl(path):
    """
    Creates semantic segmentation masks from the instance segmentation ones. It merges
    all the masks of different instances from the same image into a single one for the
    Bowl Science Kaggle dataset.

    Inputs:
        - path (str): containing the local path to the dataset.

    Outputs:

    Remarks:
    """

    names = os.listdir(os.path.join(path,'images'))
    total = len(names)

    masks = os.listdir(os.path.join(path,'masks'))
    for i in range(total):

        subnames = []
        for name in masks:
            if name.startswith(f'train_{i}_'):
                subnames.append(name)
        
        base = Image.open(os.path.join(path,'masks',subnames[0]))
        base = np.array(base)
        for name in subnames[1:]:
            mask = Image.open(os.path.join(path,'masks',name))
            array = np.array(mask)
            base += array
            
        final = base > 0
        img = Image.fromarray(final)
        img.save(f'{path}/masks/mask_{i}.png')
    
    return