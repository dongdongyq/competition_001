# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/8
"""
import os
import cv2
import argparse
import random
import tqdm
import numpy as np
from common import get_json_data, json_data_to_images_data,\
    check_or_make_dir, divide_according_sliding, save_json, make_pair


def deal_bbox(original_position, ori_bbox):
    """
    将给定的bbox坐标转换为分割后的坐标
    :param original_position: 分割后的图像原点在原图的坐标
    :param ori_bbox: 原始的bbox坐标
    :return: bbox在分割后的图片中的坐标
    """
    bboxes = []
    for x, y in original_position:
        bbox = ori_bbox
        bbox = np.array(bbox) - np.array([x, y, x, y])
        bboxes.append(bbox)
    return bboxes


def filter_bbox_and_images(bboxes, category, image_size, min_size=5):
    """
    过滤不在图像内的bbox和没有bbox的图像
    :param bboxes: 转换后的bbox的坐标
    :param category: 瑕疵类别
    :param image_size: 分割的图像大小
    :param min_size: bbox在图像边缘的最小值
    :return: 有效的bbox信息
    """
    valid_images_info = {}
    for i, bbox in enumerate(bboxes):
        if np.all(bbox < 0):
            continue
        valid_bbox = []
        valid_category = []
        for j in range(len(bbox)):
            x_min = bbox[j][0]
            x_max = bbox[j][2]
            y_min = bbox[j][1]
            y_max = bbox[j][3]
            if x_max <= min_size or y_max <= min_size:
                continue
            if x_min >= image_size[0] - min_size or y_min >= image_size[1] - min_size:
                continue
            box = np.where(bbox[j] < 0, 0, bbox[j])
            box[2:] = np.where(box[2:] > image_size, image_size, box[2:])
            valid_bbox.append(box)
            valid_category.append(category[j])
        if valid_bbox:
            valid_images_info[i] = {
                "bbox": valid_bbox,
                "category": valid_category,
            }
    return valid_images_info


def save_images(save_path, save_name, save_img):
    """
    保存图片
    :param save_path:
    :param save_name:
    :param save_img:
    :return:
    """
    path = os.path.join(save_path, save_name)
    if not os.path.exists(path):
        cv2.imwrite(path, save_img)


def generate_txt_labels(save_path, save_name, save_label):
    """
    保存yolo格式的label
    :param save_path:
    :param save_name:
    :param save_label:
    :return:
    """
    save_labels_dir = "labels"
    save_labels_path = check_or_make_dir(save_path, save_labels_dir, mkdir=True)
    save_name = save_name.split(".")[0] + ".txt"
    path = os.path.join(save_labels_path, save_name)
    if os.path.exists(path):
        return
    category = save_label["category"]
    height = save_label["height"]
    width = save_label["width"]
    bbox = save_label["bbox"]
    with open(path, "w") as fp:
        for i, box in enumerate(bbox):
            center_x = (box[0] + box[2]) / 2 / width
            center_y = (box[1] + box[3]) / 2 / height
            bw = (box[2] - box[0]) / width
            bh = (box[3] - box[1]) / height
            print(center_x, center_y, bw, bh, path)
            data = "%d %.6f %.6f %.6f %.6f\n" % (category[i] - 1, center_x, center_y, bw, bh)
            fp.write(data)


def generate_images(save_path, save_name, save_img):
    """
    将分割后的图像和label保存成yolo格式的
    :param save_path:
    :param save_name:
    :param save_img:
    :return:
    """
    save_images_dir = "images"
    save_images_path = check_or_make_dir(save_path, save_images_dir, mkdir=True)
    save_images(save_images_path, save_name, save_img)


def deal_one_image(i, nums, img_name, images_data, images_dir_path, divide_size, stride, save_path, dataset):
    """
    分割一张图像
    :param img_name: 图像名
    :param images_data: 图像label信息
    :param images_dir_path: 图像所在目录
    :param divide_size: 分割大小
    :param stride: 移动步长
    :param save_path: 保存路径
    :param dataset: save coco or voc
    :return:
    """
    image_path = os.path.join(images_dir_path, img_name)
    image_data = images_data.get(img_name, None)
    if image_data is None:
        return
    if not os.path.exists(image_path):
        return
    img = cv2.imread(image_path)
    divide_images, original_position = divide_according_sliding(img, divide_size, stride, True)
    bboxes = deal_bbox(original_position, image_data["bbox"])
    valid_images_info = filter_bbox_and_images(bboxes, image_data["category"], divide_size[::-1])
    if not valid_images_info:
        return
    pb = tqdm.tqdm(enumerate(valid_images_info.items()))
    for i, (index, item) in pb:
        pb.set_description("{}/{}".format(i, nums))
        save_name = img_name[:-4] + "_{:04}".format(i) + img_name[-4:]
        save_img = divide_images[index]
        item["height"] = divide_size[0]
        item["width"] = divide_size[1]
        generate_images(save_path, save_name, save_img, item)
        # generate_txt_labels(save_path, save_name, item)
        dataset.write(save_name, divide_size, item["bbox"], item["category"])
        # print(save_name, index, item)


class CocoData(object):
    def __init__(self, save_path, class_num=6):
        self.save_path = save_path
        self.write_json_context = self.coco_head(class_num)
        self.image_id = 0
        self.obj_id = 100000

    def write(self, name, size, boxes, categories):
        self.coco_images(name, size)
        self.coco_annotations(boxes, categories)
        self.image_id += 1

    def coco_head(self, class_num):
        categories = []
        for j in range(class_num):
            categories.append({'id': j + 1, 'name': str(j + 1), 'supercategory': 'None'})
        write_json_context = dict()
        write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2021, 'contributor': '',
                                      'date_created': '2021-01-15 11:00:08.5'}
        write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
        write_json_context['categories'] = categories
        write_json_context['images'] = []
        write_json_context['annotations'] = []
        return write_json_context

    def coco_images(self, img_name, size):
        image_context = dict()
        image_context['file_name'] = img_name
        image_context['height'] = size[0]
        image_context['width'] = size[1]
        image_context['date_captured'] = '2021-01-15 11:00:08.5'
        # 这么多id搞得我头都懵了,我猜这是第几张图序号吧,每行一张图，那当然就是第i张了
        image_context['id'] = self.image_id
        image_context['license'] = 1
        image_context['coco_url'] = ''
        image_context['flickr_url'] = ''
        self.write_json_context["images"].append(image_context)

    def coco_annotations(self, boxes, categories):
        image_boxes = []
        for i, box in enumerate(boxes):
            bbox_dict = {}
            # 我就有时候int和str不注意各种报错
            box = [round(b, 2) for b in box]
            xmin, ymin, xmax, ymax = box
            bbox_dict['id'] = self.obj_id
            bbox_dict['image_id'] = self.image_id
            bbox_dict['category_id'] = categories[i]
            bbox_dict['iscrowd'] = 0  # 前面有解释
            bbox_dict['area'] = round((xmax - xmin) * (ymax - ymin), 3)
            bbox_dict['bbox'] = box
            bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
            image_boxes.append(bbox_dict)
            self.obj_id += 1
        self.write_json_context["annotations"].extend(image_boxes)

    def __str__(self):
        for k, v in self.write_json_context.items():
            print(k, v)
        return ""

    def save(self):
        save_path = os.path.join(self.save_path, "ann.json")
        save_json(save_path, self.write_json_context)


class VocData(object):
    def __init__(self, save_path):
        self.save_path = check_or_make_dir(save_path, "Annotations", mkdir=True)
        self.categories = []

    def write(self, name, size, boxes, categories):
        save_name = name.replace(".jpg", ".xml")
        head_xml = self._voc_head(name, size)
        obj_xml = self._voc_obj(boxes, categories)
        self.save(save_name, head_xml+obj_xml)

    def _voc_head(self, name, size):
        xml_head = '''
        <annotation>
            <folder>division</folder>
            <!--文件名-->
            <filename>{}</filename>
            <source>
                <database>tile_round1 Database</database>
                <annotation>PASCAL tile_round1</annotation>
                <image>flickr</image>
                <flickrid>325991873</flickrid>
            </source>
            <owner>
                <flickrid>null</flickrid>
                <name>null</name>
            </owner>    
            <size>
                <width>{}</width>
                <height>{}</height>
                <depth>3</depth>
            </size>
            <segmented>0</segmented>
                    '''.format(name, size[1], size[0])
        return xml_head

    def _voc_obj(self, boxes, categories):
        xml_obj = '''
                <object>        
                    <name>{}</name>
                    <pose>Rear</pose>
                    <truncated>0</truncated>
                    <difficult>0</difficult>
                    <bndbox>
                        <xmin>{}</xmin>
                        <ymin>{}</ymin>
                        <xmax>{}</xmax>
                        <ymax>{}</ymax>
                    </bndbox>
                </object>
                '''
        body = ""
        for i, box in enumerate(boxes):
            # 我就有时候int和str不注意各种报错
            box = [round(b, 2) for b in box]
            xmin, ymin, xmax, ymax = box
            body += xml_obj.format(categories[i], xmin, ymin, xmax, ymax)
            if categories[i] not in self.categories:
                self.categories.append(categories[i])
        body += '''</annotation>'''
        return body

    def save(self, name, xml):
        if not name.endswith(".xml"):
            raise ValueError(name)
        save_path = os.path.join(self.save_path, name)
        with open(save_path, 'w') as fw:
            fw.write(xml)

    def save_label(self, save_path):
        with open(save_path, "w") as fp:
            for category in self.categories:
                fp.write(str(category) + "\n")


def main(args):
    save_path = args.save_path
    divide_size = make_pair(args.divide_size)
    stride = make_pair(args.stride)
    max_num = int(args.max_num)
    data_name = args.data_name
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = check_or_make_dir(save_path,
                                  "{}_size{}X{}".format(data_name, divide_size[0], divide_size[1]), mkdir=True)
    if data_name == "voc":
        dataset = VocData(save_path)
    elif data_name == "coco":
        dataset = CocoData(save_path)
    else:
        raise ValueError("not exists {}, please set voc or coco".format(data_name))

    images_dir_path = os.path.join(args.data_dir, "train_imgs")
    json_file_path = os.path.join(args.data_dir, "train_annos.json")

    images_name = os.listdir(images_dir_path)
    json_data = get_json_data(json_file_path)
    images_data = json_data_to_images_data(json_data)
    random.shuffle(images_name)
    for i, img_name in enumerate(images_name):
        if max_num != -1:
            if i >= max_num:
                continue
        deal_one_image(i, len(images_name), img_name, images_data, images_dir_path, divide_size, stride, save_path, dataset)
    if data_name == "coco":
        dataset.save()
    else:
        dataset.save_label(os.path.join(save_path, "labels.txt"))


def parse_args():
    parser = argparse.ArgumentParser(description='division dataset')
    parser.add_argument('--data_dir', default='../dataset/project_001/tile_round1', help='dataset path')
    parser.add_argument('--data_name', default='coco', help='dataset name')
    parser.add_argument('--save_path', default='../dataset/project_001/tile_round1_divide', help='save_path')
    parser.add_argument('--divide_size', nargs="+", type=int, default=[128, 128], help='size')
    parser.add_argument('--stride', nargs="+", type=int, default=[120, 120], help='stride')
    parser.add_argument('--max_num', type=int, default=100, help='if -1, save all')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print("division_image")
    main(parse_args())


