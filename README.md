## Table of contents
* [General info](#general-info)
* [Installation](#installation)
* [Outline](#outline)

# General info

*Title: 2D and 3D Analysis of Microscopy Images.*<br>
Abstract: This document collects the work during the second year master internship at the Brain Institute of Paris (ICM) associated to the M2 master studied at Sorbonne University. The main objective has been the study and analysis of 2D and 3D cell images. More precisely, the segmentation and tracking of this cells as it can give essential information about the migration and behaviour of tumors. Several models for two-classes semantic segmentation have been compared in 2D datasets and instance segmentation has been performed using Mask-RCNN. Future work aims at expand it to 3D datasets working with different object representation like point clouds or meshes instead of stack images and perform tracking on both 2D and 3D datasets.

You can find the whole document at [2D and 3D Analysis of Microscopy Images](https://lucas-rdlr.github.io/pdfs/Lucas%20Rincon%20de%20la%20Rosa%20-%20Internship%20report.pdf).

# Installation

All models used in the experiments depend on pytorch, as well as a handful of other fairly typical numerical packages. These can usually be installed manually without much trouble,
 but alternately a virtual environment file is also provided. These package versions were tested with CUDA 11.5.

A GPU with CUDA support is not mandatory, although the experiments take quite a long time even with GPU due to the weight and size of the images datasets, so it is recommended.

# Outline

 - `src` contains all the `.py` modules including datasets and training functions.
 - `notebooks` contains the `.ipynb` files with the experiments and tested functions.
 - `requirements.txt` the virtual environment file which can be used to install packages.

# Folder structure
```
├── data
│   ├── external
│   └── internal
├── experiments
├── models
├── notebooks
│   ├── data_science_bowl_2018.ipynb
│   ├── DeepLab.ipynb
│   ├── draft.ipynb
│   ├── general.ipynb
│   ├── __init__.py
│   ├── MaskRCNN.ipynb
│   ├── mesh_alice.ipynb
│   ├── prueba.py
│   ├── tracking.ipynb
│   └── UNet.ipynb
├── README.md
├── requirements.txt
├── src
│   ├── data
│   ├── datasets.py
│   ├── downloads.py
│   ├── features
│   ├── initializer.py
│   ├── __init__.py
│   ├── labeling.py
│   ├── models
│   ├── nets.py
│   ├── processing.py
│   ├── __pycache__
│   ├── segmentation.py
│   ├── training.py
│   ├── transformations.py
│   ├── visualization
│   └── visualization.py
└── tree.txt

```
