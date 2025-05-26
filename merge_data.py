import os
import shutil
import random
import glob
from pathlib import Path


def merge_and_organize_datasets(xun_dataset_path, uav_dataset_path, target_dataset_path,
                                train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    合并两个数据集，重命名文件，并分割为训练、验证和测试集

    参数:
        xun_dataset_path: 已分类的数据集路径
        uav_dataset_path: 未分类的数据集路径
        target_dataset_path: 目标数据集路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    """
    # 确保目标目录存在
    target_images_dir = os.path.join(target_dataset_path, "images")
    target_labels_dir = os.path.join(target_dataset_path, "labels")

    # 创建训练、验证和测试子目录
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(target_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(target_labels_dir, split), exist_ok=True)

    # 用于收集所有文件信息的列表
    all_files = []

    # 1. 处理第一个数据集(xun)
    print("处理第一个数据集...")
    xun_images_dir = os.path.join(xun_dataset_path, "images")
    xun_labels_dir = os.path.join(xun_dataset_path, "labels")

    # 收集训练集文件
    train_images = glob.glob(os.path.join(xun_images_dir, "train", "*.jpg"))
    for img_path in train_images:
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(xun_labels_dir, "train", f"{base_name}.txt")

        if os.path.exists(label_path):
            all_files.append({
                "img_path": img_path,
                "label_path": label_path,
                "original_name": f"xun_train_{base_name}",
                "original_split": "train"
            })

    # 收集验证集文件
    val_images = glob.glob(os.path.join(xun_images_dir, "val", "*.jpg"))
    for img_path in val_images:
        img_filename = os.path.basename(img_path)
        base_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(xun_labels_dir, "val", f"{base_name}.txt")

        if os.path.exists(label_path):
            all_files.append({
                "img_path": img_path,
                "label_path": label_path,
                "original_name": f"xun_val_{base_name}",
                "original_split": "val"
            })

    # 2. 处理第二个数据集(uav)
    print("处理第二个数据集...")
    uav_files = os.listdir(uav_dataset_path)

    # 获取所有jpg文件
    for file in uav_files:
        if file.lower().endswith('.jpg'):
            img_path = os.path.join(uav_dataset_path, file)
            base_name = os.path.splitext(file)[0]
            label_path = os.path.join(uav_dataset_path, f"{base_name}.txt")

            if os.path.exists(label_path):
                all_files.append({
                    "img_path": img_path,
                    "label_path": label_path,
                    "original_name": f"uav_{base_name}",
                    "original_split": "none"  # 这个数据集没有预先分割
                })

    # 随机打乱文件列表
    print(f"合并数据集，总文件数: {len(all_files)}")
    random.shuffle(all_files)

    # 计算各集合的样本数量
    total_samples = len(all_files)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # 分割数据集
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]

    # 重命名并复制文件到目标位置
    def process_files(file_list, split, start_idx):
        idx = start_idx
        for file_info in file_list:
            new_name = f"image_{idx:06d}"
            idx += 1

            # 复制并重命名图像文件
            dst_img = os.path.join(target_images_dir, split, f"{new_name}.jpg")
            shutil.copy2(file_info["img_path"], dst_img)

            # 复制并重命名标签文件
            dst_label = os.path.join(target_labels_dir, split, f"{new_name}.txt")
            shutil.copy2(file_info["label_path"], dst_label)

            # 更新文件名
            file_info["new_name"] = new_name
            file_info["new_split"] = split
        return idx

    # 处理各个集合
    print("开始复制文件并重命名...")
    idx = 0
    idx = process_files(train_files, "train", idx)
    idx = process_files(val_files, "val", idx)
    process_files(test_files, "test", idx)

    # 打印分割统计信息
    print(f"\n数据集合并、重命名和分割完成:")
    print(f"总样本数: {total_samples}")
    print(f"训练集: {len(train_files)} 样本 ({train_ratio * 100:.1f}%)")
    print(f"验证集: {len(val_files)} 样本 ({val_ratio * 100:.1f}%)")
    print(f"测试集: {len(test_files)} 样本 ({test_ratio * 100:.1f}%)")

    # 创建映射文件（原始名称 -> 新名称）
    mapping_file = os.path.join(target_dataset_path, "filename_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("原始数据集,原始文件名,原始分割,新文件名,新分割\n")
        for file_info in all_files:
            original_name = file_info["original_name"]
            dataset_name = "xun" if original_name.startswith("xun") else "uav"
            original_file = original_name.replace(f"{dataset_name}_", "").replace(f"{dataset_name}_train_", "").replace(
                f"{dataset_name}_val_", "")

            f.write(
                f"{dataset_name},{original_file},{file_info['original_split']},{file_info['new_name']},{file_info['new_split']}\n")

    print(f"文件名映射已保存至: {mapping_file}")


if __name__ == "__main__":
    # 定义路径
    xun_dataset_path = r"F:\project\py\pythonProject1\Graduation Project\xun"
    uav_dataset_path = r"F:\project\py\pythonProject1\Graduation Project\uav dateset\jpg_txt"
    target_dataset_path = r"F:\project\py\pythonProject1\Graduation Project\new_dataset"

    # 执行合并和分割
    merge_and_organize_datasets(xun_dataset_path, uav_dataset_path, target_dataset_path)