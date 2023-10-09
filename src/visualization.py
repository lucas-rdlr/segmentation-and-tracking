import numpy as np
import os
import tifffile as tiff
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_mini_batch(imgs, masks=None, overlay=True, batch=4):
    """
    Function for representing batches from both UNet and Mask RCNN Datasets.

    Inputs:
        - imgs: original images (one for each stack at a time).
        - masks: predicted mask from one of the models.
        - overlay: boolean indicating wether to superpose images and masks in the
                   same plots or represent them separatly.
        
    Output: figure of subplots containing always 4 colums and as many rows as necce-
            sary. Depending on the value of overlay, it will create for or less rows.
            If overlay=False, there will be double number of rows and it will alternate
            between images and masks so that the mask corresponding to an images appears
            always under it.
    """

    size = imgs.shape[0]
    rows = int(np.ceil(size/batch))
    
    if overlay:
        plt.figure(figsize=(20, 10))

        for i in range(size):
            plt.subplot(rows, batch, i+1)
            plt.imshow(imgs[i])

            if masks is not None:
                plt.imshow(masks[i], alpha=0.5)

            plt.axis('Off')

    else:
        rows *= 2
        fig, ax = plt.subplots(rows, batch, figsize=(20, np.floor(10*len(imgs)/batch)))
        for i in range(size):
            group = i // batch
            off = i % batch

            ax[2*group, off].imshow(imgs[i])
            ax[2*group, off].axis('Off')
            ax[2*group+1, off].imshow(masks[i])
            ax[2*group+1, off].axis('Off')
            
    plt.tight_layout()
    plt.show()

    return


#####################################################################


def create_video(input_path, output_path, fps):
    """
    Function to create a video from a set of ordered images.

    Inputs:
        - input_path: path to the folder containing the set of images.
        - output_path: path to export the .mp4 video.
        - fps: frames per second of the output video.
    """
    
    # Get a sorted list of image files
    image_files = sorted(os.listdir(input_path))

    # Determine the size of the first image
    first_image_path = os.path.join(input_path, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec (e.g., 'mp4v', 'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate through the image files and write each frame to the video
    for image_file in image_files:
        image_path = os.path.join(input_path, image_file)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer and close the video file
    video_writer.release()

    return


#####################################################################


def tif_to_point_cloud(file):
    """
    Function to create a point cloud image out of a stack image containing different
    2D images for different Z values.

    Input: path to the stack .tif image.

    Output: 
        - point_cloud: array generated taking as coordinates the indexes of rows and 
                       columns (as X and Y coordinates) from the image where the value
                       is greater than a threshold (0.1) and as Z coordinate the index
                       of the slice in the stack.
        - values: the value of the coordinate in the image in case we want to represent
                  the point cloud with color depending on its original value.
    """
    
    volume = tiff.imread(file) # / 255 normalize it so it is less heavy
    
    point_cloud = []
    values = []

    for i, slice in enumerate(volume):
        rows, cols = slice.shape
        z = i # Z-coordinate is the index of the slice
        
        for x in range(rows):
            for y in range(cols):

                if slice[x, y] > 0.1:  # Consider non-zero pixel values as points
                    point_cloud.append([x, y, z])
                    values.append(slice[x, y])
    
    point_cloud = np.array(point_cloud)
    values = np.array(values)

    return point_cloud, values


#####################################################################


def point_cloud_to_off(array, name, path=None):

    if path is None:
        path = os.getcwd()
    
    vertices = len(array)
    with open(f'{path}/{name}.off', 'w') as file:
        file.write('OFF\n')
        file.write(f'{vertices} 0 0\n')

        for vert in array:
            file.write(f'{vert[0]} {vert[1]} {vert[2]}\n')
        
    return


#####################################################################


def visualize_point_cloud(point_cloud, colors=None, sep=0.1):
    """
    Function to represent a point cloud giving it color depending on certein values.

    Inputs: outputs of tif_to_point_cloud(file) fucntion.
        - point_cloud: array of shape (n, 3) containing n points with (x,y,z) coordinates.
        - colors: values to color the points. By default, the height of the z coordinate will
                  be considered.
        - sep: aspect ratio to draw the z axis. 
    
    Outputs: plotly go.Figure.
    """

    if colors is None:
        colors = point_cloud[:,2]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=point_cloud[:,0], y=point_cloud[:,1], z=point_cloud[:,2], 
                mode='markers',
                marker=dict(size=1, color=colors)
            )
        ],

        layout=dict(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=sep),
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True)
            )
        )
    )

    fig.show()

    return


#####################################################################


