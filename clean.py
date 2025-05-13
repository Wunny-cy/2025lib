import os
import shutil

# 定义需要保留的类别，并为新类别创建ID映射
original_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
 20, 21, 22, 23, 24, 25, 26]  # 原始类别ID
retained_classes = [0, 2, 5, 7, 21, 24, 25, 26]  # 需要保留的类别ID

# 创建新类别ID的映射
id_mapping = {original_id: new_id for new_id, original_id in enumerate(retained_classes)}

# 设置数据集路径和目标目录路径
data_splits = ['train2012', 'val2012', 'test']
base_dir = 'E:/mine/XiangMu/LIB/2025lib/VOC2012-pre'
output_base_dir = 'ultralytics/datasets/filtered_dataset'

base_dir = os.path.abspath(base_dir)
output_base_dir = os.path.abspath(output_base_dir)
os.makedirs(output_base_dir, exist_ok=True)

for split in data_splits:
    image_dir = os.path.join(base_dir, 'images', split)
    label_dir = os.path.join(base_dir, 'labels', split)

    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Skipping {split} as it does not exist.")
        continue

    output_image_dir = os.path.join(output_base_dir, 'images', split)
    output_label_dir = os.path.join(output_base_dir, 'labels', split)

    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    # 获取所有标签文件
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        image_path = os.path.join(image_dir, label_file.replace('.txt', '.jpg'))

        with open(label_path, 'r') as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            if class_id in id_mapping:
                new_class_id = id_mapping[class_id]
                parts[0] = str(new_class_id)
                filtered_lines.append(" ".join(parts) + "\n")

        if filtered_lines:
            # 如果有保留的标签，写回新的文件
            new_label_path = os.path.join(output_label_dir, label_file)
            new_image_path = os.path.join(output_image_dir, label_file.replace('.txt', '.jpg'))

            with open(new_label_path, 'w') as f:
                f.writelines(filtered_lines)

            # 复制图像文件到新的目录
            shutil.copy(image_path, new_image_path)
        else:
            # 如果没有保留的标签，跳过此文件
            continue

print("Dataset filtering and ID remapping completed.")
