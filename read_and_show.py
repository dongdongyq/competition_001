import cv2
import json
import os
import numpy as np


dataset_path = r"D:\python\competition\dataset\tile_round1_train_20201231"
json_file = r"train_annos.json"
images_dir = r"train_imgs"


def read_json(path):
    with open(path, "r") as fp:
        data = fp.read()
        json_data = json.loads(data)
        return json_data


def draw_box(img, bbox):
    bbox = np.array(bbox, dtype=np.int)
    point_top = tuple(bbox[:2])
    point_bottom = tuple(bbox[2:])
    cv2.rectangle(img, point_top, point_bottom, (0, 0, 255), 1)
    return img


def main():
    json_path = os.path.join(dataset_path, json_file)
    json_data = read_json(json_path)
    print(type(json_data))
    # win_name = cv2.namedWindow("show")
    for data in json_data[:10]:
        print(data)
        # img_path = os.path.join(dataset_path, images_dir, data['name'])
        # show(img_path, data['bbox'])


if __name__ == '__main__':
    main()
