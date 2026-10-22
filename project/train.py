# -*- coding: utf-8 -*-
"""
自定义数据集训练脚本 —— 训练 YOLOv8 分割模型（用于越界/入侵行为监测）

核心要点：
  1. 【不使用 YOLO 自带的 COCO 预训练标签】—— 只使用 yolov8m-seg.pt 的权重作为
     初始参数(迁移学习)，在你的【自己的数据集】上训练，类别完全由你自定义
     (通常只设 person 一个类别，或你的业务类别)。
  2. 【高精度配置】—— 更大的模型(yolov8m-seg)、更大输入(imgsz=1280)、更多轮次、
     更强的数据增强、AdamW+余弦退火，训练完成后自动导出到
     runs/segment/train/weights/best.pt，main.py 会自动优先加载它。
  3. 数据集格式：YOLO 分割格式（Polygon 多边形标注），目录结构如下：

      datasets/
        data.yaml            # 数据集配置文件
        images/
          train/  xx.jpg
          val/    xx.jpg
        labels/
          train/  xx.txt     # 每行: class_id x1 y1 x2 y2 ... （多边形点，坐标归一化0~1）
          val/    xx.txt

  【标注工具】建议用 LabelMe 或 labelme2yolo 转换；也可用 Roboflow 在线标注后导出
  "YOLO v8 Segmentation" 格式。

  用法：
    python train.py                     # 使用默认 datasets/data.yaml 训练
    python train.py --data 你的yaml      # 指定数据集配置
    python train.py --epochs 300        # 指定训练轮数

  训练完成后，把 best.pt 的路径填入 config.json 的 model_path 即可被监测系统加载。
  默认最佳权重输出：runs/segment/train/weights/best.pt
"""

import argparse
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="自定义数据集训练 YOLOv8 分割模型")
    parser.add_argument("--data", default="datasets/data.yaml",
                        help="数据集配置文件路径（YOLO格式 data.yaml）")
    parser.add_argument("--model", default="yolov8m-seg.pt",
                        help="预训练权重/模型（仅作为初始参数，非COCO标签直接复用）")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮数")
    parser.add_argument("--batch", type=int, default=8, help="batch size（显存不足调小）")
    parser.add_argument("--imgsz", type=int, default=1280, help="训练输入尺寸，越大精度越高但越慢")
    parser.add_argument("--device", default="", help="如 0 或 cpu，默认自动选择")
    parser.add_argument("--resume", action="store_true",
                        help="从上次断点续训（自动读取 runs/segment/train/weights/last.pt）")
    parser.add_argument("--resume-model", default="runs/segment/train/weights/last.pt",
                        help="续训用的断点权重路径")
    parser.add_argument("--name", default="train", help="本次训练输出目录名（默认 train，可用 trainB 等避免覆盖）")
    parser.add_argument("--project", default="runs/segment", help="输出项目目录（默认 runs/segment）")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print("=" * 60)
        print("❌ 找不到数据集配置文件：", args.data)
        print("   请先准备你的自定义数据集，参考脚本顶部说明 / datasets/data.yaml 示例。")
        print("=" * 60)
        sys.exit(1)

    # 检查是否有 GPU
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if not args.device:
        args.device = 0 if has_cuda else "cpu"

    print("=" * 60)
    print("🚀 开始自定义数据集训练")
    print(f"   数据集配置 : {args.data}")
    print(f"   模型(预训练): {args.model}  ← 仅作初始权重，在你的数据上重新训练")
    print(f"   训练轮数   : {args.epochs}")
    print(f"   输入尺寸   : {args.imgsz}")
    print(f"   Batch size : {args.batch}")
    print(f"   计算设备   : {args.device}")
    print("=" * 60)

    # 从预训练权重初始化（迁移学习，非直接套用COCO标签）；若续训则从断点恢复
    if args.resume:
        if not os.path.exists(args.resume_model):
            print("❌ 未找到断点权重：", args.resume_model)
            sys.exit(1)
        print("🔄 从断点续训：", args.resume_model)
        model = YOLO(args.resume_model)
        resume_flag = True
        # 续训沿用断点保存的全部配置，只需指定数据与断点
        results = model.train(resume=True, data=args.data, device=args.device)
        _finish()
        return
    else:
        model = YOLO(args.model)
        resume_flag = False

    # 高精度训练参数
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        # ---- 优化器与学习率（收敛更稳、精度更高）----
        optimizer="AdamW",
        lr0=0.0008,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        cos_lr=True,
        # ---- 数据增强（提升泛化与精度）----
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        close_mosaic=10,
        # ---- 其他 ----
        patience=50,
        workers=4,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=42,
        val=True,
        save=True,
        save_period=-1,
        pretrained=True,
        verbose=True,
    )

    best = os.path.join(args.project, args.name, "weights", "best.pt")
    print("=" * 60)
    print("✅ 训练完成！最佳权重：", best)
    print("   请把该路径填入 config.json 的 model_path，系统将优先加载它。")
    print("=" * 60)


def _finish():
    best = os.path.join("runs", "segment", "train", "weights", "best.pt")
    print("=" * 60)
    print("✅ 训练完成！最佳权重：", best)
    print("   请把该路径填入 config.json 的 model_path，系统将优先加载它。")
    print("=" * 60)


if __name__ == "__main__":
    main()
