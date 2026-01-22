"""
部分下载 ALLaVA-4V 数据集的数据准备脚本
支持流式加载和部分处理，避免下载完整的 500GB 数据
"""
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset, config

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def get_cache_dir():
    """获取 ALLaVA-4V 缓存目录"""
    cache_dir = os.path.join(
        config.HF_DATASETS_CACHE, "FreedomIntelligence", "ALLaVA"
    )
    return cache_dir


def process_allava4v_row(row, cache_dir, available_images):
    """
    处理 ALLaVA-4V 数据行

    Args:
        row: 数据集中的一行
        cache_dir: 缓存目录
        available_images: 可用图像文件集合

    Returns:
        处理后的数据行，如果图像不存在则返回 None
    """
    conversations = row["conversations"]

    # 构建图像路径
    image_name = row.get("image", "")
    if not image_name:
        return None

    # 处理图像路径（可能是相对路径）
    if image_name.startswith("allava_laion/images/"):
        image_name = image_name.replace("allava_laion/images/", "")

    image_path = os.path.join(cache_dir, "allava_laion", "images", image_name)

    # 检查图像是否在我们下载的部分中
    if image_name not in available_images:
        return None

    if not os.path.exists(image_path):
        return None

    # 处理对话
    formatted_conversations = []
    for message in conversations:
        if message["from"] not in ROLE_MAPPING:
            continue
        new_role = ROLE_MAPPING[message["from"]]
        if new_role == "user":
            # 移除 <image> 标签
            text_content = message["value"].replace("<image>\n", "")
            content = text_content
        else:
            content = message["value"]
        formatted_conversations.append({"role": new_role, "content": content})

    return {
        "id": row["id"],
        "image": image_path,
        "conversations": formatted_conversations
    }


def get_available_images(cache_dir):
    """获取已下载的图像文件列表"""
    images_dir = os.path.join(cache_dir, "allava_laion", "images")
    if not os.path.exists(images_dir):
        return set()

    # 获取所有图像文件名
    available_images = set()
    for file in os.listdir(images_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            available_images.add(file)

    print(f"找到 {len(available_images)} 个已下载的图像")
    return available_images


def main():
    parser = argparse.ArgumentParser(description="部分准备 ALLaVA-4V 数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        default="allava4v",
        help="数据集名称（当前仅支持 allava4v）"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="输出路径，默认为 cache/dataset/"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="处理的样本数量（在过滤图像后）"
    )
    parser.add_argument(
        "--num-image-chunks",
        type=int,
        default=2,
        help="已下载的图像包数量（1-10）"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="使用流式加载（不将整个数据集加载到内存）"
    )

    args = parser.parse_args()

    if args.dataset != "allava4v":
        raise ValueError("此脚本仅支持 allava4v 数据集")

    # 获取缓存目录
    cache_dir = get_cache_dir()
    print(f"缓存目录: {cache_dir}")

    # 获取可用图像列表
    available_images = get_available_images(cache_dir)
    if not available_images:
        print("警告: 未找到任何已下载的图像！")
        print(f"请先运行: bash datasets/download_laion_partial.sh {args.num_image_chunks}")
        return

    # 设置输出路径
    if args.output_path is None:
        root_path = Path(__file__).parent.parent
        output_path = root_path.joinpath("cache", "dataset")
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path.joinpath(f"allava4v_partial_{args.num_image_chunks}chunks_train.jsonl")

    if output_file.exists():
        print(f"输出文件已存在: {output_file}")
        overwrite = input("是否覆盖？(y/n): ")
        if overwrite.lower() != 'y':
            print("已取消")
            return

    # 加载数据集
    print(f"加载 ALLaVA-4V 数据集（streaming={args.streaming}）...")
    ds = load_dataset(
        "FreedomIntelligence/ALLaVA-4V",
        name="allava_laion",
        streaming=args.streaming
    )["instruct"]

    # 处理数据
    print("处理数据...")
    processed_count = 0
    skipped_count = 0

    with open(output_file, "w") as f:
        for item in tqdm(ds, desc="处理样本"):
            # 处理数据行
            processed_row = process_allava4v_row(item, cache_dir, available_images)

            if processed_row is None:
                skipped_count += 1
                continue

            # 写入文件
            f.write(json.dumps(processed_row, ensure_ascii=False) + "\n")
            processed_count += 1

            # 检查是否达到样本数量限制
            if args.sample_size and processed_count >= args.sample_size:
                break

    print(f"\n✅ 处理完成！")
    print(f"成功处理: {processed_count} 个样本")
    print(f"跳过（图像不存在）: {skipped_count} 个样本")
    print(f"输出文件: {output_file}")
    print(f"\n下一步: 修改训练脚本中的 --train-data-path 为:")
    print(f"  {output_file}")


if __name__ == "__main__":
    main()
