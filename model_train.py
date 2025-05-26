from ultralytics import YOLO
import os
import torch # 导入 torch 检查环境

# --- 检查环境 ---
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前 GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("警告：未能检测到 CUDA GPU，将在 CPU 上运行（非常慢！）")


# !!! 修改为指向你上次运行结果中的 best.pt !!!
# 例如: /root/zscloud-tmp/yolo/runs_ultralytics/uav_yolov8s_final_run/weights/best.pt
model_weights = '/root/zscloud-tmp/yolo/runs_ultralytics/uav_yolov8s_final_run/weights/best.pt' # <--- 修改这里，指向 best.pt

# 2. 数据集配置文件路径 (!!! 确认这个路径是正确的 !!!)
data_yaml = '/root/zscloud-tmp/yolo/new_dataset/new.yaml'

# 3. 训练超参数 (为第二次微调调整)
epochs = 100       # 微调轮数 (根据需要调整，可以先设少点，比如 50-100)
batch_size = 32    # 尝试减小批次大小 (可选，如果显存允许，也可以保持 64)
img_size = 640     # 输入图片尺寸 (保持不变)
workers = 8        # 数据加载进程数 (保持不变)
project_name = '/root/zscloud-tmp/yolo/runs_ultralytics' # 保存结果的主目录
run_name = 'uav_yolov8s_finetune_run' # <--- 新的训练名称，区分第一次
device = 0         # 使用 GPU 0

# --- 检查权重和数据配置文件路径 ---
if not os.path.exists(model_weights):
     print(f"错误: 权重文件不存在 {model_weights}")
     exit()
else:
     print(f"加载最佳权重进行微调: {model_weights}")

if not os.path.exists(data_yaml):
     print(f"错误: 数据集配置文件不存在 {data_yaml}")
     exit()
else:
    print(f"使用数据集配置文件: {data_yaml}")

# --- 加载模型 ---
# 直接加载 best.pt 权重进行微调
model = YOLO(model_weights)

# --- 开始微调训练 ---
print(f"\n === 开始微调训练 ===")
print(f"  模型: {model_weights}")
print(f"  数据集: {data_yaml}")
print(f"  微调 Epochs: {epochs}")
print(f"  Batch Size: {batch_size}") # 注意 batch size 可能已更改
print(f"  加载 'best.pt' 权重开始...")
print(f"  图像尺寸: {img_size}")
print(f"  设备: GPU {device}")
print(f"  结果将保存在: {project_name}/{run_name}")

# --- 启动训练 ---
results = model.train(
    data=data_yaml,
    epochs=epochs,
    patience=30,      # 可以适当减少早停轮数 (例如 30)
    batch=batch_size,
    imgsz=img_size,
    device=device,
    workers=workers,
    project=project_name,
    name=run_name,
    exist_ok=False,   # 设置为 False 确保不会覆盖同名文件夹，除非你确定要覆盖
    amp=True,         # 保持混合精度训练

    # --- 微调关键参数 ---
    resume=False,     # 设置为 False，因为我们是开始新的微调阶段，而不是恢复之前的训练状态
    lr0=0.001,        # <--- 降低初始学习率 (例如从默认的 0.01 降到 0.001)
    lrf=0.001,        # <--- 降低最终学习率因子 (例如从默认的 0.01 降到 0.001 或 0.0001)

    # --- 增加数据增强 (示例，根据需要调整强度) ---
    hsv_h=0.020,      # 色调增强范围 (默认 0.015)
    hsv_s=0.75,       # 饱和度增强范围 (默认 0.7)
    hsv_v=0.45,       # 亮度增强范围 (默认 0.4)
    degrees=5.0,      # 旋转角度范围 (默认 0.0)
    translate=0.15,   # 平移范围 (默认 0.1)
    scale=0.6,        # 缩放范围 (默认 0.5)
    shear=2.0,        # 剪切角度范围 (默认 0.0)
    perspective=0.0005,# 透视变换系数 (默认 0.0)
    flipud=0.1,       # 上下翻转概率 (默认 0.0)
    fliplr=0.5,       # 左右翻转概率 (默认 0.5)
    mosaic=0.8,       # Mosaic增强概率 (默认 1.0，可以稍微降低或保持)
    mixup=0.1,        # MixUp增强概率 (默认 0.0，可以尝试开启)
    copy_paste=0.1    # Copy-Paste增强概率 (默认 0.0，可以尝试开启)
)

print("\n === 微调训练完成! ===")
print(f"训练结果和模型保存在: {results.save_dir}") # results.save_dir 是实际保存的路径

# (可选) 训练完成后自动在验证集上评估最佳模型
print("\n === 开始评估最佳模型在验证集上的性能 ===")
# 加载训练过程中保存的最佳权重
model_best = YOLO(os.path.join(results.save_dir, 'weights/best.pt'))
metrics = model_best.val()
print("评估指标 (mAP50-95):", metrics.box.map)