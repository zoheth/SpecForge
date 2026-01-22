"""
创建最小的多模态验证数据集
用于快速验证 Qwen VL EAGLE3 训练流程，无需下载大量数据
"""
import json
import os
from pathlib import Path
import argparse

# 简单的多模态问答模板
QA_TEMPLATES = [
    ("描述这张图片", "这是一张展示了各种视觉元素的图片。"),
    ("这张图片中有什么？", "图片中包含了多个对象和场景元素。"),
    ("详细说明图片内容", "图片展示了丰富的视觉信息，包括多个主要元素。"),
    ("图片的主要内容是什么？", "图片的主要内容集中在展示的核心对象上。"),
    ("请分析这张图片", "这张图片通过视觉元素传达了特定的信息。"),
    ("图片中最显眼的是什么？", "图片中最显眼的是位于中心的主要对象。"),
    ("这张图片想表达什么？", "这张图片旨在展示特定的场景或概念。"),
    ("用一句话总结图片", "这是一张包含多个元素的综合性图片。"),
]

def download_sample_images(image_dir, num_images=100):
    """
    下载示例图像用于测试
    这里使用 COCO val2017 的一小部分作为示例
    """
    from datasets import load_dataset

    print(f"下载 {num_images} 张示例图像...")

    # 使用 HuggingFace datasets 的 COCO 数据集
    # 这是一个小得多的选择
    try:
        ds = load_dataset("HuggingFaceM4/COCO", split="validation", streaming=True)

        image_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i, item in enumerate(ds):
            if count >= num_images:
                break

            # 保存图像
            image = item['image']
            image_path = image_dir / f"sample_{count:04d}.jpg"
            image.save(str(image_path))
            count += 1

            if count % 10 == 0:
                print(f"已下载 {count}/{num_images} 张图像")

        print(f"✅ 成功下载 {count} 张图像到 {image_dir}")
        return count

    except Exception as e:
        print(f"警告: 自动下载图像失败: {e}")
        print("请手动准备图像文件到:", image_dir)
        return 0


def create_dataset(image_dir, output_file, num_samples=500, auto_download=False, num_download=100):
    """
    创建最小验证数据集

    Args:
        image_dir: 图像目录
        output_file: 输出 jsonl 文件路径
        num_samples: 生成的样本数量
        auto_download: 是否自动下载示例图像
        num_download: 自动下载的图像数量
    """
    image_dir = Path(image_dir)

    # 如果需要自动下载且目录为空
    if auto_download:
        existing_images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        if len(existing_images) < 10:
            download_sample_images(image_dir, num_download)

    # 获取所有图像
    image_files = list(image_dir.glob("*.jpg")) + \
                  list(image_dir.glob("*.png")) + \
                  list(image_dir.glob("*.jpeg"))

    if not image_files:
        raise ValueError(f"未在 {image_dir} 中找到图像文件！请先准备图像或使用 --auto-download")

    print(f"找到 {len(image_files)} 张图像")

    # 生成数据
    data = []
    for i in range(num_samples):
        # 循环使用图像和问答模板
        img_idx = i % len(image_files)
        qa_idx = (i // len(image_files)) % len(QA_TEMPLATES)

        question, answer = QA_TEMPLATES[qa_idx]

        # 为每个问题添加一些变化
        sample_id = f"minimal_{i:05d}"

        data.append({
            "id": sample_id,
            "image": str(image_files[img_idx].absolute()),
            "conversations": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        })

    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 成功生成 {len(data)} 个样本到 {output_path}")
    print(f"📊 数据统计:")
    print(f"  - 图像数量: {len(image_files)}")
    print(f"  - 样本数量: {len(data)}")
    print(f"  - 每张图像重复使用: {len(data) // len(image_files)} 次")
    print(f"\n📝 下一步:")
    print(f"  1. 修改训练脚本中的 --train-data-path 为:")
    print(f"     {output_path}")
    print(f"  2. 设置较少的训练轮数，例如: --num-epochs 2")
    print(f"  3. 开始训练!")


def main():
    parser = argparse.ArgumentParser(description="创建最小的多模态验证数据集")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="cache/custom_vl_data/images",
        help="图像目录路径"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="cache/dataset/minimal_vl_train.jsonl",
        help="输出文件路径"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="生成的样本数量"
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="自动下载示例图像（从 COCO 数据集）"
    )
    parser.add_argument(
        "--num-download",
        type=int,
        default=100,
        help="自动下载的图像数量"
    )

    args = parser.parse_args()

    create_dataset(
        image_dir=args.image_dir,
        output_file=args.output_file,
        num_samples=args.num_samples,
        auto_download=args.auto_download,
        num_download=args.num_download
    )


if __name__ == "__main__":
    main()
