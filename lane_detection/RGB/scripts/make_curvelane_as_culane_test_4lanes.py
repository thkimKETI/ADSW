import os
import cv2
import tqdm
import json
import numpy as np
import json, argparse
import imagesize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='The root of the dataset')
    return parser


def calc_k(line, height, width, angle=False):
    '''
    Calculate the direction of lanes
    '''
    line_x = line[::2]
    line_y = line[1::2]

    length = np.sqrt((line_x[0]-line_x[-1])**2 + (line_y[0]-line_y[-1])**2)
    if length < 90:
        return -10
    p = np.polyfit(line_x, line_y, deg = 1)
    rad = np.arctan(p[0])

    if angle:
        return rad

    try:
        curve = np.polyfit(line_x[:2], line_y[:2], deg = 1)
    except Exception:
        curve = np.polyfit(line_x[:3], line_y[:3], deg = 1)

    try:
        curve1 = np.polyfit(line_y[:2], line_x[:2], deg = 1)
    except Exception:
        curve1 = np.polyfit(line_y[:3], line_x[:3], deg = 1)

    if rad < 0:
        y = np.poly1d(curve)(0)
        if y > height:
            result = np.poly1d(curve1)(height)
        else:
            result = -(height-y)
    else:
        y = np.poly1d(curve)(width)
        if y > height:
            result = np.poly1d(curve1)(height)
        else:
            result = width+(height-y)
    
    # print(line_x)
    # print(line_y)

    return result


def read_label(label_path):
    f = open(label_path, 'r')
    lines = json.load(f)['Lines']
    f.close()
    line_txt = []
    for line in lines:
        temp_line = []
        line = sorted(line, key=lambda x: -float(x['y']))
        for point in line:
            temp_line.append(float(point['x']))
            temp_line.append(float(point['y']))
        line_txt.append(temp_line)

    return line_txt


def get_4_lanes_label(lines, width, height, x_factor, y_factor):

    all_lanes = []
    lane_pt_list = []
    ks = []
    ks_theta = []

    for line in lines:
        ks_line = calc_k(line, height, width)                # get the direction of each lane
        ks_theta_line = calc_k(line, height, width, angle=True)           # get the direction of each lane (return radian)
        ks.append(ks_line)
        ks_theta.append(ks_theta_line)
        line = np.array(line)
        line[::2] = line[::2] * x_factor
        line[1::2] = line[1::2] * y_factor
        lane_pt_list.append(line)
        #lane_pt.append(list(zip(line_x, line_y)))

    ks = list(enumerate(ks))
    ks_new = [] # [[index, ks, k_theta], ...]
    for idx, k in enumerate(ks):
        k = list(k)
        k.append(ks_theta[idx])
        ks_new.append(k)

    k_neg = []
    k_pos = []

    for k in ks_new:
        if k[2] == -10: # -10 means the lane is too short and is discarded
            continue
        elif k[2] < 0:
            k_neg.append(k)
        elif k[2] > 0:
            k_pos.append(k)
    # k_neg = ks[ks_theta<0].copy()
    # k_neg_theta = ks_theta[ks_theta<0].copy()
    # k_pos = ks[ks_theta>0].copy()
    # k_pos_theta = ks_theta[ks_theta>0].copy()
    # k_neg = k_neg[k_neg_theta != -10]                                      # -10 means the lane is too short and is discarded
    # k_pos = k_pos[k_pos_theta != -10]
    k_neg.sort(key = lambda x:x[1], reverse = True)
    k_pos.sort(key = lambda x:x[1])

    for idx in range(len(k_neg))[:2]:
        pt_idx = k_neg[idx][0]
        all_lanes.append(lane_pt_list[pt_idx])
        
    for idx in range(len(k_pos))[:2]:
        pt_idx = k_pos[idx][0]
        all_lanes.append(lane_pt_list[pt_idx])
        
    return all_lanes


def generate_linestxt_on_curvelane_val():
    args = get_args().parse_args()
    curvelane_val_root = os.path.join(args.root, 'valid')

    assert os.path.exists(curvelane_val_root)

    assert os.path.exists(os.path.join(curvelane_val_root, 'images'))

    list_file = os.path.join(curvelane_val_root, 'valid.txt')

    all_files = open(list_file, 'r').readlines()

    for file in tqdm.tqdm(all_files):
        file = file.strip()
        label_path = file.replace('images', 'labels')
        label_path = label_path.replace('.jpg', '.lines.json')

        label_path = os.path.join(curvelane_val_root, label_path)
        file_path = os.path.join(curvelane_val_root, file)

        width, height = imagesize.get(file_path)
        x_factor = 2560 / width
        y_factor = 1440 / height

        lines = read_label(label_path)
        culane_style_label = get_4_lanes_label(lines, width, height, x_factor, y_factor)
        culane_style_label_store_path = os.path.join(curvelane_val_root, file).replace('jpg','lines.txt')
        with open(culane_style_label_store_path, 'w') as f:
            for culane_style_label_i in culane_style_label:
                for pt in culane_style_label_i:
                    pt = round(pt,3)
                    f.write(str(pt)+' ')
                f.write('\n')


    fp = open(os.path.join(curvelane_val_root, 'valid.txt'), 'r')
    res = fp.readlines()
    fp.close()
    res = [os.path.join('valid', r) for r in res]
    with open(os.path.join(curvelane_val_root, 'valid_for_culane_style.txt'), 'w') as fp:
        fp.writelines(res)


if __name__ == "__main__":
    generate_linestxt_on_curvelane_val()
