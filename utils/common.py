# -*- coding: utf-8 -*-
"""
author: DongYanQiang
data: 2021/1/15
"""
import os
import json
import cv2
import numpy as np
import subprocess
import logging

COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
          (0, 255, 255), (255, 0, 255), (255, 255, 0), (255, 10, 100)]


def execute_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while p.poll() is None:
        line = p.stdout.readline().decode("utf-8")
        line = line.strip()
        if line:
            print('{} command output: {}'.format(cmd, line))
    if p.returncode == 0:
        logging.info("success execute " + cmd)
        return True
    else:
        logging.error("fail execute " + cmd)
        return False


def show(img, win_name="img", wait=0):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(wait)


def get_json_data(file_path):
    if not os.path.exists(file_path):
        print("Error ", file_path)

    with open(file_path, "r") as fp:
        return json.loads(fp.read())


def json_data_to_images_data(json_data):
    """
    将给定的label转换成以图片名为key的字典
    :param json_data:
    :return:
    """
    images_data = {}
    for item in json_data:
        if item["name"] not in images_data:
            images_data[item["name"]] = {
                "name": item["name"],
                'height': item["image_height"],
                'width': item["image_width"],
                'category': [item["category"]],
                'bbox': [item["bbox"]],
            }
        else:
            images_data[item["name"]]["category"].append(item["category"])
            images_data[item["name"]]["bbox"].append(item["bbox"])
    return images_data


def check_or_make_dir(root, dir_name, mkdir=False):
    dir_path = os.path.join(root, dir_name)
    if not os.path.exists(dir_path):
        if mkdir:
            os.makedirs(dir_path)
        else:
            raise ValueError("Error path: " + dir_path)
    return dir_path


def save_json(save_path, result):
    if not save_path.endswith(".json"):
        raise ValueError(save_path)
    with open(save_path, 'w') as fp:
        json.dump(result, fp, indent=4, ensure_ascii=False)


def xywh_to_xyxy(box):
    box = np.array(box, dtype=np.float)
    xy = box[:2]
    wh = box[2:]
    box[:2] = xy - wh//2
    box[2:] = xy + wh//2
    return list(box)


def divide_according_sliding(image, size, stride, padding=False):
    """
    通过滑动窗口切分图片
    :param image: 要切分的图片
    :param size: 切分后图片的大小
    :param stride: 滑动步长
    :param padding: 边缘填充
    :return: 切分后的图片列表
    """
    ori_size = image.shape[:2]
    size = np.array(size)
    stride = np.array(stride)
    # 计算分割后的图像数量
    num = (ori_size - size) // stride + 1
    if padding:
        num += 1
        pad = (num - 1) * stride + size - ori_size
        image = np.pad(image, ((0, pad[0]), (0, pad[1]), (0, 0)), 'constant', constant_values=0)
    else:
        crop = ori_size - (num - 1) * stride - size
        image = image[:-crop[0], :-crop[1], :]
    division_images = []
    original_position = []
    for i in range(num[0]):
        for j in range(num[1]):
            start = np.array([i, j]) * stride
            end = start + size
            crop_img = image[start[0]:end[0], start[1]:end[1], :]
            division_images.append(crop_img)
            original_position.append(list(start[::-1]))
    return division_images, original_position


def divide_according_point(image, size, boxes):
    """
    通过指定点的方式分割图片
    :param image: 要分割的图像
    :param size: 分割后的图像大小
    :param boxes: 瑕疵box list
    :return: 分割后的图像列表
    """
    ori_size = image.shape[:2]
    size = np.array(size)
    num = len(boxes)
    division_images = []
    for i in range(num):
        box = np.array(boxes[i], dtype=np.int)
        point = np.array(box[:2], dtype=np.int)
        box_size = box[2:] - box[:2]
        high = np.where(size - box_size <= 10, size, size - box_size)
        crop_y = np.random.randint(0, high[0])
        crop_x = np.random.randint(0, high[1])
        crop_lt = np.array([crop_x, crop_y])
        crop_lt = point - crop_lt
        crop_lt = np.where(crop_lt < 0, 0, crop_lt)
        crop_tl = crop_lt[::-1]
        crop_br = crop_tl + size
        crop_br = np.where(crop_br > ori_size, ori_size, crop_br)
        crop_tl = crop_br - size
        crop = np.append(crop_tl, crop_br)
        print(ori_size, crop)
        crop_img = image[crop[0]:crop[2], crop[1]:crop[3], :]
        division_images.append(crop_img)
    return division_images


def make_pair(obj):
    if isinstance(obj, int):
        return obj, obj
    elif isinstance(obj, list) or isinstance(obj, tuple):
        if len(obj) == 1:
            return int(obj[0]), int(obj[0])
        return [int(o) for o in obj]
    elif isinstance(obj, str):
        return int(obj), int(obj)
    else:
        raise ValueError(obj)


def draw_box(img, boxes, names, scores=None):
    if scores is None:
        scores = [1. for _ in range(len(boxes))]
    for i, box in enumerate(boxes):
        p1 = int(box[0]), int(box[1])
        p2 = int(box[2]), int(box[3])
        if scores[i] == 1.:
            color = (50, 100, 200)
        else:
            color = COLORS[int(names[i])]
        cv2.rectangle(img, p1, p2, color, 1)
        cv2.putText(img, str(names[i]) + " {:.3f}".format(scores[i]), p1, cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)


class CocoData(object):
    def __init__(self, save_path, class_num=6):
        self.save_path = save_path
        self.write_json_context = self.coco_head(class_num)
        self.image_id = 0
        self.obj_id = 100000

    def write(self, name, size, boxes, categories, segmentation=None):
        self.coco_images(name, size)
        self.coco_annotations(boxes, categories, segmentation)
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

    def coco_annotations(self, boxes, categories, segmentation=None):
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
            if segmentation is None:
                bbox_dict['segmentation'] = [[xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]]
            else:
                segmentation[0] = [int(s) for s in segmentation[0]]
                bbox_dict['segmentation'] = segmentation
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
    def __init__(self, save_path, save_seg=False):
        self.save_path = save_path
        self.categories = []
        self.save_seg = save_seg

    def write(self, name, size, boxes, categories):
        save_name = name.replace(".jpg", ".xml")
        head_xml = self._voc_head(name, size)
        obj_xml = self._voc_obj(boxes, categories)
        self.save_xml(save_name, head_xml+obj_xml)
        if self.save_seg:
            seg_save_name = name.replace(".jpg", ".png")
            self.save_segmentation(seg_save_name, size, boxes, categories)

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

    def save_xml(self, name, xml):
        save_path = check_or_make_dir(self.save_path, "Annotations", mkdir=True)
        if not name.endswith(".xml"):
            raise ValueError(name)
        save_path = os.path.join(save_path, name)
        with open(save_path, 'w') as fw:
            fw.write(xml)

    def save_segmentation(self, save_name, size, boxes, categories):
        mask = np.zeros(size, dtype=np.uint8)
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            contour = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax], dtype=np.int).reshape(-1, 2)
            cv2.drawContours(mask, [contour], 0, 255, -1)
        save_path = check_or_make_dir(self.save_path, "Segmentation", mkdir=True)
        save_path = os.path.join(save_path, save_name)
        cv2.imwrite(save_path, mask)

    def save(self):
        save_path = os.path.join(self.save_path, "labels.txt")
        with open(save_path, "w") as fp:
            for category in self.categories:
                fp.write(str(category) + "\n")
        if self.save_seg:
            seg_save_path = os.path.join(self.save_path, "seg_labels.txt")
            with open(seg_save_path, "w") as fp:
                for i in range(2):
                    fp.write(str(i) + "\n")


if __name__ == '__main__':
    print("common")
