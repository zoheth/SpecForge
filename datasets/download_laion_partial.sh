#!/bin/bash
# 部分下载 ALLaVA-4V 数据集用于快速验证
# 用法: bash download_laion_partial.sh [num_chunks]
# 例如: bash download_laion_partial.sh 2  # 只下载前2个图像包（约100GB）

# 默认下载2个图像包（约占总数据集的20%）
NUM_CHUNKS=${1:-2}

if [ "$NUM_CHUNKS" -gt 10 ]; then
    echo "错误: 最多只有10个图像包，您指定了 $NUM_CHUNKS"
    exit 1
fi

echo "将下载前 $NUM_CHUNKS 个图像包（总共10个）"
echo "预计下载大小: 约 $((NUM_CHUNKS * 50))GB"

laion_root="allava_laion"

mkdir -p $laion_root
cd $laion_root

# 1. 下载标注文件
echo "步骤 1/3: 下载标注文件..."
## 1.1 caption
if [ ! -f "ALLaVA-Caption-LAION-4V.json" ]; then
    wget -c -O ALLaVA-Caption-LAION-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/ALLaVA-Caption-LAION-4V.json?download=true
else
    echo "Caption 文件已存在，跳过"
fi

## 1.2 instruction
if [ ! -f "ALLaVA-Instruct-LAION-4V.json" ]; then
    wget -c -O ALLaVA-Instruct-LAION-4V.json https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/ALLaVA-Instruct-LAION-4V.json?download=true
else
    echo "Instruct 文件已存在，跳过"
fi

# 2. 下载和解压图像（只下载指定数量）
mkdir -p image_chunks
mkdir -p images/

echo "步骤 2/3: 下载前 $NUM_CHUNKS 个图像包..."
## 2.1 下载
for ((i=0; i<$NUM_CHUNKS; i++))
do
    if [ ! -f "image_chunks/images_$i.zip" ]; then
        echo "下载 images_$i.zip ..."
        wget -c -O image_chunks/images_$i.zip https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/resolve/main/allava_laion/image_chunks/images_$i.zip?download=true &
    else
        echo "images_$i.zip 已存在，跳过"
    fi
done

wait

echo "步骤 3/3: 解压图像文件..."
## 2.2 解压
for ((i=0; i<$NUM_CHUNKS; i++))
do
    echo "解压 images_$i.zip ..."
    unzip -j -o image_chunks/images_$i.zip -d images/ &
done

wait
echo ""
echo "✅ 部分下载完成！"
echo "已下载: $NUM_CHUNKS / 10 个图像包"
echo "下一步: 运行 python scripts/prepare_data_partial.py --dataset allava4v --num-image-chunks $NUM_CHUNKS"
