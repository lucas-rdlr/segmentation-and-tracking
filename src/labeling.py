import numpy as np
import json
from PIL import Image, ImageDraw

def readJson(file_path, img_size=(512, 443), box=False):
    """
    Function to create a mask out of json file output by labelme GUI.
    Other GUI may also work but changes should be made.

    Inputs:
        - file_path (str): path to the json file.
        - img_size (tuple): size of the mask we want to create. Should be of the
                same size of the original image segmented.
        - box (bool): indicating wether to create also a detection box
               around the object or just the segmentation mask.
    
    Output: PIL Image containing the mask.
    """

    # Load the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Access the annotation data
    shapes = data['shapes']

    # Create a new figure and axes
    img = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(img)

    # Create segmentation mask (out of polygons coordinates)
    if not box:
        for i, shape in enumerate(shapes):
            points = shape['points']
            coords = np.array(points).reshape(-1).astype(np.float32)
            draw.polygon(coords, fill=i+1)
        
        return img

    # Create box detection out of max/min coordinates
    for i, shape in enumerate(shapes):
        points = shape['points']
        x = np.sort([points[0][0],points[1][0]])
        y = np.sort([points[0][1],points[1][1]])

        coords = np.array((x,y)).T.reshape(-1).astype(np.float32)
        draw.rectangle(coords, outline=i+1)
    
    return img


#####################################################################