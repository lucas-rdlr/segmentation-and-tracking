import numpy as np
import torch


#####################################################################


class CenterCrop(object):

    def __init__(self, output_size=None):
        
        if output_size is None:
            self.output_size = None
        
        elif isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        
        elif isinstance(output_size, (int, tuple)) and len(output_size) == 2:
            self.output_size = output_size

        else:
            assert False, "Input data type is not valid"

    def __call__(self, list):

        image = list[0]

        h, w = image.shape[:2]

        if self.output_size is not None:
            new_h, new_w = self.output_size
        
        else:
            factor = np.floor(min(h,w)/16)
            new_size = np.random.randint(1, factor+1) * 16
            new_h, new_w = new_size, new_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2
        
        new_list = []
        for elem in list:
            elem_transformed = elem[top: top + new_h,
                                    left: left + new_w]
            new_list.append(elem_transformed)

        return new_list
    

#####################################################################


class RandomCrop(object):

    def __init__(self, output_size=None):

        if output_size is None:
            self.output_size = None
        
        elif isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        
        elif isinstance(output_size, (int, tuple)) and len(output_size) == 2:
            self.output_size = output_size

        else:
            assert False, "Input data type is not valid"

    def __call__(self, list):

        image = list[0]
        
        h, w = image.shape[:2]

        if self.output_size is not None:
            new_h, new_w = self.output_size
        
        else:
            factor = np.floor(min(h,w)/16)
            new_size = np.random.randint(12, factor+1) * 16
            new_h, new_w = new_size, new_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        new_list = []
        for elem in list:
            elem_transformed = elem[top: top + new_h,
                                    left: left + new_w]
            new_list.append(elem_transformed)

        return new_list
    

#####################################################################


class VerticalFlip(object):

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, list):

        p = np.random.rand()

        new_list = []
        for elem in list:

            if p < self.probability:
                elem = np.flip(elem, 0)

            new_list.append(elem)

        return new_list


#####################################################################


class HorizontalFlip(object):

    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, list):

        p = np.random.rand()

        new_list = []
        for elem in list:

            if p < self.probability:
                elem = np.flip(elem, 1)
                
            new_list.append(elem)

        return new_list
    

#####################################################################


class ToTensor(object):

    def __call__(self, list):

        new_list = []
        for elem in list:
            
            elem = elem.copy()
            elemTensor = torch.from_numpy(elem)
            new_list.append(elemTensor)

        return new_list