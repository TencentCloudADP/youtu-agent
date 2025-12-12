#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单的CNN训练MNIST数据集全流程
包含：数据加载、模型定义、训练、验证、测试、模型保存
"""

import argparse
import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# MNIST类别名称
CLASS_NAMES = [str(i) for i in range(10)]


def setup_logging(output_dir: str, log_name: str = "training"):
    """
    配置logging模块

    Args:
        output_dir: 日志输出目录
        log_name: 日志文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"{log_name}_{timestamp}.log")

    # 配置logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存至: {log_path}")

    return logger, log_path


class SimpleCNN(nn.Module):
    """简单的CNN模型用于MNIST分类"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        # 第二个卷积块: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))
        # 第三个卷积块: 7x7 -> 3x3
        x = self.pool(F.relu(self.conv3(x)))

        # 展平
        x = x.view(-1, 128 * 3 * 3)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def get_data_loaders(data_dir: str, batch_size: int, num_workers: int = 4):
    """
    获取MNIST数据加载器

    Args:
        data_dir: 数据存储目录
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        train_loader, test_loader
    """
    logger = logging.getLogger(__name__)

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])

    # 加载训练集
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # 加载测试集
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")

    return train_loader, test_loader


def compute_class_metrics(all_preds, all_targets, num_classes=10):
    """
    计算每个类别的精度、召回率和F1分数

    Args:
        all_preds: 所有预测结果
        all_targets: 所有真实标签
        num_classes: 类别数量

    Returns:
        metrics_dict: 包含每个类别指标的字典
    """
    metrics = {}

    for cls in range(num_classes):
        # True Positives: 预测为cls且实际为cls
        tp = ((all_preds == cls) & (all_targets == cls)).sum().item()
        # False Positives: 预测为cls但实际不是cls
        fp = ((all_preds == cls) & (all_targets != cls)).sum().item()
        # False Negatives: 预测不是cls但实际是cls
        fn = ((all_preds != cls) & (all_targets == cls)).sum().item()

        # 计算精度 (Precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # 计算召回率 (Recall)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': (all_targets == cls).sum().item()  # 该类别的样本数
        }

    return metrics


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    logger = logging.getLogger(__name__)
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []

    # pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        batch_loss = loss.item()
        total_loss += batch_loss
        batch_losses.append(batch_loss)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # 更新进度条
        # pbar.set_postfix({
        #     'loss': f'{batch_loss:.4f}',
        #     'acc': f'{100. * correct / total:.2f}%'
        # })

        # 每100个batch记录一次
        if (batch_idx + 1) % 100 == 0:
            logger.debug(f"Epoch {epoch} Batch {batch_idx + 1}/{len(train_loader)} | "
                         f"Loss: {batch_loss:.4f} | Acc: {100. * correct / total:.2f}%")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    logger.info(f"[Train] Epoch {epoch} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy


def validate(model, test_loader, criterion, device, desc="Val", detailed=False):
    """
    验证/测试模型

    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
        desc: 描述
        detailed: 是否输出详细的每类别指标
    """
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        # pbar = tqdm(test_loader, desc=f"[{desc}]", leave=False)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # 收集所有预测和标签用于计算每类别指标
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    # 合并所有预测和标签
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # 计算每个类别的指标
    class_metrics = compute_class_metrics(all_preds, all_targets)

    # 记录总体结果
    logger.info(f"[{desc}] Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% ({correct}/{total})")

    if detailed:
        # 输出每个类别的详细指标
        logger.info("=" * 70)
        logger.info(f"{'类别':<10} {'精度(P)':<12} {'召回(R)':<12} {'F1分数':<12} {'样本数':<10}")
        logger.info("-" * 70)

        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

        for cls in range(10):
            m = class_metrics[cls]
            logger.info(f"{CLASS_NAMES[cls]:<10} {m['precision']:.4f}       {m['recall']:.4f}       "
                        f"{m['f1']:.4f}       {m['support']:<10}")
            macro_precision += m['precision']
            macro_recall += m['recall']
            macro_f1 += m['f1']

        # 计算宏平均
        macro_precision /= 10
        macro_recall /= 10
        macro_f1 /= 10

        logger.info("-" * 70)
        logger.info(f"{'宏平均':<10} {macro_precision:.4f}       {macro_recall:.4f}       {macro_f1:.4f}")
        logger.info("=" * 70)

    return avg_loss, accuracy, class_metrics


def save_checkpoint(model, optimizer, epoch, accuracy, checkpoint_path):
    """保存模型检查点"""
    logger = logging.getLogger(__name__)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"模型已保存至: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="CNN训练MNIST分类模型")

    # 位置参数
    parser.add_argument("--output_dir", default="./pgs/output", type=str, help="输出目录，用于保存模型和日志")

    # 可选参数
    parser.add_argument("--data-dir", type=str, default="./pgs/data", help="MNIST数据存储目录 (默认: ./pgs/data)")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数 (默认: 10)")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小 (默认: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率 (默认: 0.001)")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数 (默认: 4)")
    parser.add_argument("--device", type=str, default="auto", help="设备: auto/cuda/cpu (默认: auto)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")

    args = parser.parse_args()

    # 设置logging
    logger, log_path = setup_logging(args.output_dir)

    # 记录训练配置
    logger.info("=" * 60)
    logger.info("训练配置")
    logger.info("=" * 60)
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 设置设备
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"使用设备: {device}")

    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建模型
    model = SimpleCNN().to(device)
    logger.info(f"\n模型结构:\n{model}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )

    # 训练循环
    best_accuracy = 0.0
    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # 验证 (每个epoch结束时显示详细指标)
        val_loss, val_acc, _ = validate(
            model, test_loader, criterion, device, desc="Val", detailed=(epoch % 5 == 0 or epoch == args.epochs)
        )

        # 调整学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            logger.info(f"学习率调整: {current_lr:.6f} -> {new_lr:.6f}")

        # 打印epoch总结
        logger.info(f"Epoch {epoch:3d}/{args.epochs} 完成 | "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, val_acc, best_model_path)
            logger.info(f"新的最佳模型! 准确率: {val_acc:.2f}%")

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model.pth")
    save_checkpoint(model, optimizer, args.epochs, val_acc, final_model_path)

    # 最终测试
    logger.info("\n" + "=" * 60)
    logger.info("最终测试 (使用最佳模型)")
    logger.info("=" * 60)

    # 加载最佳模型进行测试
    checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, class_metrics = validate(
        model, test_loader, criterion, device, desc="Test", detailed=True
    )

    logger.info(f"\n最终测试结果:")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_acc:.2f}%")
    logger.info(f"  最佳验证准确率: {best_accuracy:.2f}% (Epoch {checkpoint['epoch']})")

    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info(f"模型保存位置: {args.output_dir}")
    logger.info(f"日志保存位置: {log_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
