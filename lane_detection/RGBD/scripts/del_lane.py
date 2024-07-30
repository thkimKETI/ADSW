import os
import cv2
import tqdm
import json
import numpy as np
import json, argparse
import imagesize
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the CurveLanes dataset')
    return parser

args = get_args().parse_args()
curvelane_val_root = os.path.join(args.root, 'valid')

assert os.path.exists(curvelane_val_root)

assert os.path.exists(os.path.join(curvelane_val_root, 'images'))

annot_folder = os.path.join(curvelane_val_root, 'images')

gt_file = os.path.join(curvelane_val_root, 'train_gt.txt')

list_file = os.path.join(curvelane_val_root, 'valid.txt')

all_files = open(gt_file, 'r').readlines()

for file in tqdm.tqdm(all_files):
    file = file.strip()
    file = file.replace('train', 'valid')
    img_file=file[13:49]
    img_annot_file=img_file.replace('.jpg','.lines.txt')
    exist_lane = file[-7:]

    lines = open(os.path.join(annot_folder, img_annot_file), 'r').readlines()

    if exist_lane=='1 1 1 1':
        del lines[4:]
    elif exist_lane=='0 1 1 1' or exist_lane=='1 1 1 0':
        del lines[3:]
    elif exist_lane=='0 1 1 0':
        del lines[2:]

    final_file = os.path.join(annot_folder, img_annot_file)
    f = open(final_file, 'w')
    for i in range(len(lines)):
        f.write(lines[i])

    f.close()
