import os
import cv2
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

# python yolo2coco.py --root_dir VisDrone2019-DET-train --save_path train.json
# python yolo2coco.py --root_dir VisDrone2019-DET-val --save_path val.json
# python yolo2coco.py --root_dir VisDrone2019-DET-test-dev --save_path test.json

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', default='./dataset/valid', type=str,
                    help="root path of images and labels, include ./images and ./labels and classes.txt")
parser.add_argument('--save_path', type=str, default='./valid.json',
                    help="if not split the dataset, give a path to a json file")
parser.add_argument('--random_split', action='store_true', help="random split the dataset, default ratio is 8:1:1")
parser.add_argument('--split_by_file', action='store_true',
                    help="define how to split the dataset, include ./train.txt ./val.txt ./test.txt ")

arg = parser.parse_args()


def train_test_val_split_random(img_paths, ratio_train=0.8, ratio_test=0.1, ratio_val=0.1):
    # 这里可以修改数据集划分的比例。
    assert int(ratio_train + ratio_test + ratio_val) == 1
    train_img, middle_img = train_test_split(img_paths, test_size=1 - ratio_train, random_state=233)
    ratio = ratio_val / (1 - ratio_train)
    val_img, test_img = train_test_split(middle_img, test_size=ratio, random_state=233)
    print("NUMS of train:val:test = {}:{}:{}".format(len(train_img), len(val_img), len(test_img)))
    return train_img, val_img, test_img


def train_test_val_split_by_files(root_dir):
    # 根据文件 train.txt, val.txt, test.txt（里面写的都是对应集合的图片名字） 来定义训练集、验证集和测试集
    phases = ['train', 'val', 'test']
    img_splits = []

    for p in phases:
        define_path = os.path.join(root_dir, f'{p}.txt')
        print(f'Read {p} dataset definition from {define_path}')
        assert os.path.exists(define_path), f"找不到文件: {define_path}"

        with open(define_path, 'r') as f:
            # 读取每行并去除空白字符
            img_paths = [line.strip() for line in f.readlines() if line.strip()]
            print(f"在{p}.txt中找到{len(img_paths)}个图片")
            # 如果需要的话，这里可以打印几个样本检查格式
            if img_paths:
                print(f"示例图片名称: {img_paths[:3]}")
            img_splits.append(img_paths)

    return img_splits[0], img_splits[1], img_splits[2]


def yolo2coco(arg):
    root_path = arg.root_dir
    print("Loading data from ", root_path)

    assert os.path.exists(root_path), f"根目录不存在: {root_path}"

    originLabelsDir = os.path.join(root_path, 'labels')
    originImagesDir = os.path.join(root_path, 'images')

    assert os.path.exists(originImagesDir), f"图片目录不存在: {originImagesDir}"
    assert os.path.exists(originLabelsDir), f"标签目录不存在: {originLabelsDir}"

    # 读取类别文件
    classes_path = os.path.join(root_path, 'classes.txt')
    assert os.path.exists(classes_path), f"类别文件不存在: {classes_path}"

    with open(classes_path) as f:
        classes = f.read().strip().split()

    print(f"找到的类别: {classes}")

    # 初始化数据集
    train_dataset = {'categories': [], 'annotations': [], 'images': []}
    val_dataset = {'categories': [], 'annotations': [], 'images': []}
    test_dataset = {'categories': [], 'annotations': [], 'images': []}
    dataset = {'categories': [], 'annotations': [], 'images': []}

    # 建立类别标签和数字id的对应关系, 类别id从1开始
    for i, cls in enumerate(classes, 1):
        category = {'id': i, 'name': cls, 'supercategory': 'mark'}
        train_dataset['categories'].append(category)
        val_dataset['categories'].append(category)
        test_dataset['categories'].append(category)
        dataset['categories'].append(category)

    # 标注的id和图片id计数器
    ann_id_cnt = 0
    img_id_cnt = 0

    if arg.split_by_file:
        print("使用文件定义的划分模式")
        try:
            train_img_list, val_img_list, test_img_list = train_test_val_split_by_files(root_path)
        except Exception as e:
            print(f"读取划分文件出错: {e}")
            return

        # 检查images目录中的子目录结构
        subdirs = ['train', 'val', 'test']
        for subdir in subdirs:
            subdir_path = os.path.join(originImagesDir, subdir)
            if not os.path.exists(subdir_path):
                print(f"警告: 图片子目录不存在: {subdir_path}")
                print(f"创建目录: {subdir_path}")
                os.makedirs(subdir_path)

        # 处理每个子目录
        for subdir in subdirs:
            img_list = None
            current_dataset = None

            if subdir == 'train':
                img_list = train_img_list
                current_dataset = train_dataset
            elif subdir == 'val':
                img_list = val_img_list
                current_dataset = val_dataset
            elif subdir == 'test':
                img_list = test_img_list
                current_dataset = test_dataset

            images_subdir = os.path.join(originImagesDir, subdir)
            labels_subdir = os.path.join(originLabelsDir, subdir)

            if not os.path.exists(labels_subdir):
                print(f"警告: 标签子目录不存在: {labels_subdir}")
                print(f"创建目录: {labels_subdir}")
                os.makedirs(labels_subdir)

            print(f"处理 {subdir} 集合中的 {len(img_list)} 个图片")

            # 处理每个图片文件
            for img_file in tqdm(img_list, desc=f"处理{subdir}"):
                # 构建图片完整路径
                img_path = os.path.join(images_subdir, img_file)

                if not os.path.exists(img_path):
                    print(f"警告: 图片不存在: {img_path}")
                    continue

                # 读取图像
                try:
                    im = cv2.imread(img_path)
                    if im is None:
                        print(f"错误: 无法读取图像: {img_path}")
                        continue

                    height, width, _ = im.shape
                except Exception as e:
                    print(f"读取图像时出错: {img_path}, 错误: {e}")
                    continue

                # 添加图像信息
                current_dataset['images'].append({
                    'file_name': img_file,
                    'id': img_id_cnt,
                    'width': width,
                    'height': height
                })

                # 构建对应的标签文件路径
                label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(labels_subdir, label_file)

                if not os.path.exists(label_path):
                    print(f"警告: 标签文件不存在: {label_path}")
                    img_id_cnt += 1
                    continue

                # 读取标签
                with open(label_path, 'r') as fr:
                    label_lines = fr.readlines()
                    for label in label_lines:
                        label = label.strip().split()
                        if len(label) < 5:
                            print(f"警告: 标签格式不正确: {label}")
                            continue

                        try:
                            cls_id = int(label[0])
                            x = float(label[1])
                            y = float(label[2])
                            w = float(label[3])
                            h = float(label[4])
                        except ValueError:
                            print(f"警告: 无法解析标签: {label}")
                            continue

                        # 转换坐标
                        x1 = (x - w / 2) * width
                        y1 = (y - h / 2) * height
                        x2 = (x + w / 2) * width
                        y2 = (y + h / 2) * height

                        # 转换为COCO格式的类别ID (从1开始)
                        coco_cls_id = cls_id + 1

                        bbox_width = max(0, x2 - x1)
                        bbox_height = max(0, y2 - y1)

                        # 添加标注信息
                        current_dataset['annotations'].append({
                            'area': bbox_width * bbox_height,
                            'bbox': [x1, y1, bbox_width, bbox_height],
                            'category_id': coco_cls_id,
                            'id': ann_id_cnt,
                            'image_id': img_id_cnt,
                            'iscrowd': 0,
                            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                        })
                        ann_id_cnt += 1

                img_id_cnt += 1

    elif arg.random_split:
        # 实现随机分割的逻辑
        print("随机分割模式")

        # 获取所有图片文件
        all_images = []
        for subdir in ['train', 'val', 'test']:
            subdir_path = os.path.join(originImagesDir, subdir)
            if os.path.exists(subdir_path):
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith(('.jpg', '.png')):
                        all_images.append((subdir, img_file))

        print(f"总共找到 {len(all_images)} 张图片")

        # 随机划分
        train_split, rest = train_test_split(all_images, test_size=0.2, random_state=233)
        val_split, test_split = train_test_split(rest, test_size=0.5, random_state=233)

        print(f"随机划分结果: 训练集 {len(train_split)}, 验证集 {len(val_split)}, 测试集 {len(test_split)}")

        # 处理每个划分
        splits = [(train_split, train_dataset), (val_split, val_dataset), (test_split, test_dataset)]

        for split_images, current_dataset in splits:
            for subdir, img_file in tqdm(split_images):
                # 构建图片完整路径
                img_path = os.path.join(originImagesDir, subdir, img_file)

                if not os.path.exists(img_path):
                    print(f"警告: 图片不存在: {img_path}")
                    continue

                # 读取图像
                try:
                    im = cv2.imread(img_path)
                    if im is None:
                        print(f"错误: 无法读取图像: {img_path}")
                        continue

                    height, width, _ = im.shape
                except Exception as e:
                    print(f"读取图像时出错: {img_path}, 错误: {e}")
                    continue

                # 添加图像信息
                current_dataset['images'].append({
                    'file_name': img_file,
                    'id': img_id_cnt,
                    'width': width,
                    'height': height
                })

                # 构建对应的标签文件路径
                label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
                label_path = os.path.join(originLabelsDir, subdir, label_file)

                if not os.path.exists(label_path):
                    print(f"警告: 标签文件不存在: {label_path}")
                    img_id_cnt += 1
                    continue

                # 读取标签
                with open(label_path, 'r') as fr:
                    label_lines = fr.readlines()
                    for label in label_lines:
                        label = label.strip().split()
                        if len(label) < 5:
                            print(f"警告: 标签格式不正确: {label}")
                            continue

                        try:
                            cls_id = int(label[0])
                            x = float(label[1])
                            y = float(label[2])
                            w = float(label[3])
                            h = float(label[4])
                        except ValueError:
                            print(f"警告: 无法解析标签: {label}")
                            continue

                        # 转换坐标
                        x1 = (x - w / 2) * width
                        y1 = (y - h / 2) * height
                        x2 = (x + w / 2) * width
                        y2 = (y + h / 2) * height

                        # 转换为COCO格式的类别ID (从1开始)
                        coco_cls_id = cls_id + 1

                        bbox_width = max(0, x2 - x1)
                        bbox_height = max(0, y2 - y1)

                        # 添加标注信息
                        current_dataset['annotations'].append({
                            'area': bbox_width * bbox_height,
                            'bbox': [x1, y1, bbox_width, bbox_height],
                            'category_id': coco_cls_id,
                            'id': ann_id_cnt,
                            'image_id': img_id_cnt,
                            'iscrowd': 0,
                            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                        })
                        ann_id_cnt += 1

                img_id_cnt += 1

    else:
        # 不分割的情况
        print("不分割数据集")

        # 获取所有图片
        all_images = []
        for subdir in ['train', 'val', 'test']:
            subdir_path = os.path.join(originImagesDir, subdir)
            if os.path.exists(subdir_path):
                for img_file in os.listdir(subdir_path):
                    if img_file.endswith(('.jpg', '.png')):
                        all_images.append((subdir, img_file))

        print(f"总共找到 {len(all_images)} 张图片")

        for subdir, img_file in tqdm(all_images):
            # 构建图片完整路径
            img_path = os.path.join(originImagesDir, subdir, img_file)

            if not os.path.exists(img_path):
                print(f"警告: 图片不存在: {img_path}")
                continue

            # 读取图像
            try:
                im = cv2.imread(img_path)
                if im is None:
                    print(f"错误: 无法读取图像: {img_path}")
                    continue

                height, width, _ = im.shape
            except Exception as e:
                print(f"读取图像时出错: {img_path}, 错误: {e}")
                continue

            # 添加图像信息
            dataset['images'].append({
                'file_name': img_file,
                'id': img_id_cnt,
                'width': width,
                'height': height
            })

            # 构建对应的标签文件路径
            label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(originLabelsDir, subdir, label_file)

            if not os.path.exists(label_path):
                print(f"警告: 标签文件不存在: {label_path}")
                img_id_cnt += 1
                continue

            # 读取标签
            with open(label_path, 'r') as fr:
                label_lines = fr.readlines()
                for label in label_lines:
                    label = label.strip().split()
                    if len(label) < 5:
                        print(f"警告: 标签格式不正确: {label}")
                        continue

                    try:
                        cls_id = int(label[0])
                        x = float(label[1])
                        y = float(label[2])
                        w = float(label[3])
                        h = float(label[4])
                    except ValueError:
                        print(f"警告: 无法解析标签: {label}")
                        continue

                    # 转换坐标
                    x1 = (x - w / 2) * width
                    y1 = (y - h / 2) * height
                    x2 = (x + w / 2) * width
                    y2 = (y + h / 2) * height

                    # 转换为COCO格式的类别ID (从1开始)
                    coco_cls_id = cls_id + 1

                    bbox_width = max(0, x2 - x1)
                    bbox_height = max(0, y2 - y1)

                    # 添加标注信息
                    dataset['annotations'].append({
                        'area': bbox_width * bbox_height,
                        'bbox': [x1, y1, bbox_width, bbox_height],
                        'category_id': coco_cls_id,
                        'id': ann_id_cnt,
                        'image_id': img_id_cnt,
                        'iscrowd': 0,
                        'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                    })
                    ann_id_cnt += 1

            img_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)

    if arg.random_split or arg.split_by_file:
        for phase, data in zip(['train', 'val', 'test'], [train_dataset, val_dataset, test_dataset]):
            json_name = os.path.join(folder, f'{phase}.json')
            print(f"数据集 {phase} 包含 {len(data['images'])} 张图片和 {len(data['annotations'])} 个标注")
            with open(json_name, 'w') as f:
                json.dump(data, f)
            print(f'保存标注到 {json_name}')
    else:
        json_name = os.path.join(folder, arg.save_path)
        print(f"数据集包含 {len(dataset['images'])} 张图片和 {len(dataset['annotations'])} 个标注")
        with open(json_name, 'w') as f:
            json.dump(dataset, f)
        print(f'保存标注到 {json_name}')


if __name__ == "__main__":
    yolo2coco(arg)