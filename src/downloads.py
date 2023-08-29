import os
import requests
import zipfile
import tifffile as tiff
from PIL import Image

from pathlib import Path


def download_data(cell_type=None):
    """
    Download 2D and 3D datasets from http://celltrackingchallenge.net/ 

    Input (str): name of dataset to download.
    Output: creates a folder in data/external/Cell Challenge with the training and challenge .zip files.
    """

    twoD_types = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7",
                  "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa", "PhC-C2DH-U373",
                  "PhC-C2DL-PSC", "Fluo-N2DH-SIM+"]

    threeD_types = ["Fluo-C3DH-A549", "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N3DH-CE",
                    "Fluo-N3DH-CHO", "Fluo-C3DH-A549-SIM", "Fluo-N3DH-SIM+"]

    if cell_type == None:
        print("Possible datasets to download:")

        print("2D datasets:")
        print(twoD_types)

        print("3D datasets:")
        print(threeD_types)

        return

    if cell_type not in twoD_types + threeD_types:
        print('The dataset your are looking for does not exist. Call download_data() for a list os possible cell types.')
        
        return
    
    if cell_type == '2D':
        for cell in twoD_types:
            download_data(cell)
    
    if cell_type == '3D':
        for cell in threeD_types:
            download_data(cell)
    
    if cell_type == 'All':
        for cell_two in twoD_types:
            download_data(cell_two)
        
        for cell_three in threeD_types:
            download_data(cell_three)

    trainingdata_url = 'http://data.celltrackingchallenge.net/training-datasets/'
    testdata_url = 'http://data.celltrackingchallenge.net/test-datasets/'

    urls = [trainingdata_url, testdata_url]
    folders = ['training_data', 'test_data']

    path = 'data/external/Cell Challenge'

    if not os.path.isdir(path):
        os.makedirs(path)

    for i in range(2):
        folder = f'{path}/{folders[i]}'
        file = f'{path}/{folders[i]}/{cell_type}.zip'

        if not os.path.isdir(folder):
            os.makedirs(folder)

        if os.path.isfile(file):
            print(f'{cell_type} {folders[i]} data already downloaded')
            pass

        else:
            print(f'Getting url request for {cell_type} {folders[i]} ...')
            url = f'{urls[i]}{cell_type}.zip'
            r = requests.get(url)

            try:
                print(f'Downloading {folders[i]} to {folder} ...')
                with open(file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except:
                print('There was an error attepting to connect to the server')
    
    return


#####################################################################


def unzip_data(cell_type):
    """
    Unzip files previously downloaded with download_data(cell_type)

    Input (str): name of dataset to unzip.
    Output: unzips the training and challenge files in data/external/Cell Challenge if they are
            already downloaded but not unzipped. If not, returns an error.
    """

    twoD_types = ["BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7",
                  "Fluo-C2DL-MSC", "Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa", "PhC-C2DH-U373",
                  "PhC-C2DL-PSC", "Fluo-N2DH-SIM+"]

    threeD_types = ["Fluo-C3DH-A549", "Fluo-C3DH-H157", "Fluo-C3DL-MDA231", "Fluo-N3DH-CE",
                    "Fluo-N3DH-CHO", "Fluo-C3DH-A549-SIM", "Fluo-N3DH-SIM+"]

    if cell_type == None:
        print("Possible datasets to unzip:")

        print("2D datasets:")
        print(twoD_types)

        print("3D datasets:")
        print(threeD_types)

        return

    if cell_type not in twoD_types + threeD_types and cell_type != 'All':
        print('The dataset your are looking for does not exist. Call download_data() for a list os possible cell types.')
        
        return

    path = 'data/external/Cell Challenge'
    folders = ['training_data', 'test_data']
    
    if cell_type == 'All':
        disp = os.listdir(f"{path}/training_data")

        for cells in disp:
            cells = cells.split(sep='.')[0]
            unzip_data(cells)

    for folder in folders:
        file = f'{path}/{folder}/{cell_type}'

        if not os.path.isfile(f'{file}.zip'):
            print(f'{cell_type}/{folder}.zip does not exist. Try downloading it first')
            continue

        if os.path.isdir(file):
            print(f'{cell_type} {folder} data already unzipped')
            continue

        print(f'Unzipping {cell_type} {folder}...')
        with zipfile.ZipFile(f'{file}.zip', 'r') as z:
            z.extractall(f'{path}/{folder}')
          
    return


#####################################################################


def download_software():

    evalsoftware_url = 'http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip'
    evalsoftware_path = f'/data/external/Cell Challenge/evaluation_software'

    if os.path.isdir(evalsoftware_path):
        print('Sotfware already downloaded')
    
    else:
        print(f'Downloading evaluation software to {evalsoftware_path} ...')
        os.makedirs(evalsoftware_path)
        r = requests.get(evalsoftware_url)
        file = f'{evalsoftware_path}/EvaluationSoftware.zip'
        with open(file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print('Unzipping evaluation software ...')
        with zipfile.ZipFile(file, 'r') as z:
            z.extractall(evalsoftware_path)
    
    return


#####################################################################


def unstack_tiff(path_in, path_out):
    """
    The function takes a series of .tiff files composed of stacked 3d images and separates
    each of them in the z axis into 2d images as .jpg files.

    Inputs:
        - path_in (str): path to the folder containing the .tiff files.
        - path_out (str): path to the folder to save the .jpg images.

    Remarks: the number of .jpg files in the output folder will be n*m where n is the number
             of .tiff files in the input folder and m the number of slices (in the z axis)
             per file.
    """

    if not os.path.isdir(path_out):
        os.makedirs(path_out)

    tifs = os.listdir(path_in)

    for tif in tifs:
        arr = tiff.imread(os.path.join(path_in,tif))
        
        for idx,ele in enumerate(arr):
            img = Image.fromarray(ele)
            name = tif.split('.')
            img.save(f'{path_out}/{name[0]}_{idx}.jpg')