## Table of contents
* [General info](#general-info)
* [Instalation](#instalation)
* [Outline](#outline)

# General info

This repository contains the code for the experiments and algorithms developed during the Internship project at Paris Brain Instiute (ICM) which is described in the document above.

# Instalation

All models used in the expeirments depend on pytorch, as well as a handful of other fairly typical numerical packages. These can usually be installed manually without much trouble,
 but alternately a virtual environment file is also provided. These package versions were tested with CUDA 11.5.

A GPU with CUDA support is not mandatory, although the experiments take quite a long time even with GPU due to the weight and size of the images datasets, so it is recommended.

# Outline

 - `src` contatins all the `.py` modules including datasets and training functions.
 - `notebooks` contains the `.ipynb` files with the experiments and tested functions.
 - `environment.yml` the virtual environment file which can be used to install packages.
