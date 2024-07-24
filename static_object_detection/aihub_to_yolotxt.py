import json
import os
import glob

def load_json(json_path):
    with open(json_path) as f:
        json_object = json.load(f) 
    return json_object 

def get_img_info(json_object):
    f_name = json_object['image']['filename']
    size = json_object['image']['imsize']
    width, height = size[0], size[1]
    return f_name, width, height

def TL_class(red, yellow, green, left_arrow, x_light, others_arrow):
    if red == 'on' and yellow == 'off' and green == 'off' and left_arrow == 'off' and x_light == 'off' and others_arrow == 'off':
        return 0
    elif red == 'off' and yellow == 'on' and green == 'off' and left_arrow == 'off' and x_light == 'off' and others_arrow == 'off':
        return 1
    elif red == 'off' and yellow == 'off' and green == 'on' and left_arrow == 'off' and x_light == 'off' and others_arrow == 'off':
        return 2
    elif red == 'on' and yellow == 'off' and green == 'off' and left_arrow == 'on' and x_light == 'off' and others_arrow == 'off':
        return 3 
    elif red == 'on' and yellow == 'on' and green == 'off' and left_arrow == 'off' and x_light == 'off' and others_arrow == 'off':
        return 4 
    elif red == 'off' and yellow == 'off' and green == 'on' and left_arrow == 'on' and x_light == 'off' and others_arrow == 'off':
        return 5
    elif x_light == 'on':
        return 6
    elif others_arrow == 'on':
        return 7
    else:
        return 8 

def bbox_to_yolobox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

    

def get_anno_info(json_object):
    annotations = json_object['annotation']
    f_name, width, height = get_img_info(json_object)
    result = [] 

    for index in range(len(annotations)):
        annot = annotations[index]
        if (annot['class'] == 'traffic_light') and (annot['type'] == 'car'):
            red = annot['attribute'][0]['red']
            yellow = annot['attribute'][0]['yellow']
            green = annot['attribute'][0]['green']
            left_arrow = annot['attribute'][0]['left_arrow']
            x_light = annot['attribute'][0]['x_light'] 
            others_arrow = annot['attribute'][0]['others_arrow']
            CLS_ID = TL_class(red, yellow, green, left_arrow, x_light, others_arrow) 
            bbox = annot['box']
            YOLOBOX = bbox_to_yolobox(bbox, width, height)
            YOLOBOX_string = " ".join([str(x) for x in YOLOBOX])
            result.append(f"{CLS_ID} {YOLOBOX_string}")

    if result:
        with open(os.path.join(output_dir, f"{f_name[:-4]}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result))
    return 0

input_dir = "./"
image_dir = "/data/AD2_DB/Training/bb/images/"
output_dir = "./txt_labels/"
os.mkdir(output_dir) 

files = glob.glob(os.path.join(input_dir, '*.json'))
for fil in files:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    # check if the label contains the corresponding image file
    if not os.path.exists(os.path.join(image_dir, f"{filename[:-7]}.jpg")):
        print(f"{filename} image does not exist!")
        continue

    json_obj = load_json(fil)
    get_anno_info(json_obj)
    print(fil)
