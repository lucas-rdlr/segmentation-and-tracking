import os
import numpy as np
from PIL import Image

def processWarwick(path):

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


def processBowl(path):

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


def maskBowk(path):

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