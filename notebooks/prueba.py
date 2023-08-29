import sys
sys.path.append('/media/user/T7 Touch/Stage_Lucas/segmentation/')  # Replace with the actual path

import argparse
from src.downloads import download_data, unzip_data

parser = argparse.ArgumentParser()
parser.add_argument("--cell_type", type=str, default = None)
parser.add_argument("--unzip", type=str, default = None)
args = parser.parse_args()

cell_type = args.cell_type
unzip_types = args.unzip

download_data(cell_type)
unzip_data(unzip_types)