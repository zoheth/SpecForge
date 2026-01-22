# Qwen VL EAGLE3 快速验证指南（部分数据）

本指南提供多种方法，让您可以在**不下载完整 500GB 数据集**的情况下快速验证 Qwen VL EAGLE3 训练流程。

## 📊 数据规模对比

| 方案 | 数据量 | 下载大小 | 训练时间（1 epoch） | 推荐场景 |
|------|--------|---------|-------------------|---------|
| **方案1：部分 ALLaVA** | ~20k 样本 | ~100GB（2个图像包） | ~2-3小时 | 完整流程验证 |
| **方案2：最小 ALLaVA** | ~10k 样本 | ~50GB（1个图像包） | ~1-2小时 | 快速测试 |
| **方案3：流式加载** | 1k-10k 样本 | 按需下载 | ~30分钟-2小时 | 最省存储 |
| **方案4：自定义数据** | 100-1000 样本 | <1GB | ~10-30分钟 | 概念验证 |
| 完整 ALLaVA（参考） | ~100k 样本 | ~500GB | ~1-2天 | 正式训练 |

---

## 🚀 方案1：部分下载 ALLaVA（推荐）

### 优点
- ✅ 数据量足够验证完整训练流程
- ✅ 数据质量与完整数据集一致
- ✅ 可以观察到明显的训练效果

### 步骤

#### 1. 部分下载图像数据

ALLaVA-4V 数据集的图像被分成了 **10 个压缩包**（每个约 50GB），我们只需下载前面几个：

```bash
# 进入数据集目录
cd SpecForge/datasets

# 下载前 2 个图像包（约 100GB，包含约 20% 的数据）
bash download_laion_partial.sh 2

# 或者只下载 1 个（约 50GB，包含约 10% 的数据）
bash download_laion_partial.sh 1
```

**下载时间估算**：
- 1 个图像包：30分钟 - 2小时（取决于网络速度）
- 2 个图像包：1-4小时

#### 2. 处理部分数据

```bash
# 返回项目根目录
cd ..

# 处理已下载的部分数据
python scripts/prepare_data_partial.py \
    --dataset allava4v \
    --num-image-chunks 2 \
    --sample-size 5000  # 可选：进一步限制样本数量
```

**输出**：
- 生成文件：`cache/dataset/allava4v_partial_2chunks_train.jsonl`
- 包含约 20k 样本（如果指定 `--sample-size 5000` 则为 5k 样本）

#### 3. 修改训练脚本

复制官方训练脚本并修改数据路径：

```bash
# 复制官方脚本
cp examples/run_qwen2.5_7b_vl_eagle3_online.sh examples/run_qwen2.5_7b_vl_eagle3_quick.sh

# 编辑脚本，修改以下行：
# --train-data-path cache/dataset/allava4v_train.jsonl
# 改为：
# --train-data-path cache/dataset/allava4v_partial_2chunks_train.jsonl
#
# 同时建议减少训练轮数用于快速验证：
# --num-epochs 2  # 或者 3
```

#### 4. 开始训练

```bash
bash examples/run_qwen2.5_7b_vl_eagle3_quick.sh 1
```

---

## ⚡ 方案2：流式加载（最省存储）

### 优点
- ✅ 几乎不占用本地存储
- ✅ 按需下载数据
- ✅ 适合快速实验

### 缺点
- ⚠️ 需要稳定的网络连接
- ⚠️ 训练速度可能受网络影响

### 步骤

#### 1. 使用流式加载脚本

```bash
python scripts/prepare_data_partial.py \
    --dataset allava4v \
    --streaming \
    --sample-size 1000 \
    --output-path cache/dataset/
```

这会：
1. 流式加载数据集（不下载完整数据）
2. 只处理前 1000 个**有图像**的样本
3. 图像会自动按需下载到缓存

#### 2. 训练

修改训练脚本的数据路径后启动训练（同方案1）。

**注意**：流式模式下，图像可能分散在 HuggingFace 缓存中，需要确保网络连接稳定。

---

## 🎨 方案3：创建最小自定义数据集

### 优点
- ✅ 完全控制数据内容
- ✅ 最小存储占用（<1GB）
- ✅ 最快的验证速度

### 适用场景
- 概念验证
- 调试代码
- 理解数据格式

### 步骤

#### 1. 准备少量图像

```bash
# 创建数据目录
mkdir -p cache/custom_vl_data/images

# 下载或复制 100-1000 张图像到该目录
# 例如，从互联网下载一些示例图像
```

#### 2. 创建数据集脚本

创建文件 `scripts/create_minimal_vl_dataset.py`：

```python
import json
import os
from pathlib import Path

# 配置
IMAGE_DIR = "cache/custom_vl_data/images"
OUTPUT_FILE = "cache/dataset/minimal_vl_train.jsonl"
NUM_SAMPLES = 500  # 生成 500 个样本

# 获取所有图像
image_files = list(Path(IMAGE_DIR).glob("*.jpg")) + \
              list(Path(IMAGE_DIR).glob("*.png"))

print(f"找到 {len(image_files)} 张图像")

# 简单的问答模板
qa_templates = [
    ("描述这张图片", "这是一张图片，包含..."),
    ("这张图片中有什么？", "图片中包含..."),
    ("详细说明图片内容", "图片展示了..."),
]

# 生成数据
data = []
for i in range(min(NUM_SAMPLES, len(image_files) * len(qa_templates))):
    img_idx = i % len(image_files)
    qa_idx = (i // len(image_files)) % len(qa_templates)

    question, answer = qa_templates[qa_idx]

    data.append({
        "id": f"custom_{i:05d}",
        "image": str(image_files[img_idx].absolute()),
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    })

# 保存
os.makedirs("cache/dataset", exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ 生成 {len(data)} 个样本到 {OUTPUT_FILE}")
```

#### 3. 运行脚本

```bash
python scripts/create_minimal_vl_dataset.py
```

#### 4. 训练

修改训练脚本：
```bash
--train-data-path cache/dataset/minimal_vl_train.jsonl \
--num-epochs 3 \
--batch-size 2
```

**注意**：这个方案主要用于**流程验证**，由于数据量太小且质量一般，训练出的模型可能性能较差。

---

## 🔍 方案4：使用其他小规模多模态数据集

### 方案 4A：LLaVA-Instruct-150K

LLaVA-Instruct 数据集较小（约 15GB），适合快速验证：

```python
# 创建 scripts/prepare_llava_instruct.py
from datasets import load_dataset
import json
from pathlib import Path

# 加载数据集
ds = load_dataset("liuhaotian/LLaVA-Instruct-150K")["train"]

# 下载 COCO 图像（需要约 15GB）
# 参考：https://cocodataset.org/#download

output_path = Path("cache/dataset/llava_instruct_train.jsonl")

processed = []
for item in ds:
    # 处理数据格式...
    # 注意：需要适配 SpecForge 的格式
    pass

# 保存
```

### 方案 4B：使用文本数据集暂时替代

如果只是想验证 EAGLE3 **训练流程**（不包括多模态部分），可以先用文本数据：

```bash
# 使用 UltraChat（纯文本，但可以快速验证训练流程）
python scripts/prepare_data.py --dataset ultrachat --sample-size 10000

# 修改训练脚本，移除 --is-vlm 等多模态参数
# 使用纯文本配置：configs/qwen2.5-7b-eagle3.json（如果存在）
```

**注意**：这个方案会**失去多模态训练**的部分，但可以快速验证 EAGLE3 的核心训练逻辑。

---

## 📋 推荐的快速验证流程

根据您的目标选择合适的方案：

### 场景1：我想快速看到训练跑起来（30分钟内）

```bash
# 1. 创建最小数据集（方案3）
python scripts/create_minimal_vl_dataset.py

# 2. 修改训练脚本，设置：
#    --train-data-path cache/dataset/minimal_vl_train.jsonl
#    --num-epochs 2
#    --batch-size 2

# 3. 启动训练
bash examples/run_qwen2.5_7b_vl_eagle3_quick.sh 1
```

### 场景2：我想用真实数据验证完整流程（半天内）

```bash
# 1. 部分下载 ALLaVA（方案1）
cd datasets && bash download_laion_partial.sh 1

# 2. 处理数据
cd .. && python scripts/prepare_data_partial.py \
    --dataset allava4v \
    --num-image-chunks 1 \
    --sample-size 5000

# 3. 训练 2-3 个 epoch
# 修改脚本后启动
bash examples/run_qwen2.5_7b_vl_eagle3_quick.sh 1
```

### 场景3：我想要接近真实的训练体验（1-2天）

```bash
# 1. 下载更多数据（方案1）
cd datasets && bash download_laion_partial.sh 3

# 2. 处理数据（不限制 sample-size）
cd .. && python scripts/prepare_data_partial.py \
    --dataset allava4v \
    --num-image-chunks 3

# 3. 完整训练（10 epochs）
bash examples/run_qwen2.5_7b_vl_eagle3_quick.sh 2
```

---

## ⚠️ 常见问题

### Q1: 部分数据训练的模型质量如何？

**A**:
- **1-2个图像包（10-20% 数据）**：可以观察到训练效果，但模型质量会明显低于完整训练
- **3-5个图像包（30-50% 数据）**：质量接近完整训练，适合大部分验证场景
- **建议**：快速验证用少量数据，正式训练用完整数据

### Q2: 如何知道下载了哪些图像？

**A**: 检查缓存目录：

```bash
# 查看已下载的图像包
ls -lh ~/.cache/huggingface/datasets/FreedomIntelligence/ALLaVA/allava_laion/image_chunks/

# 查看解压的图像数量
ls ~/.cache/huggingface/datasets/FreedomIntelligence/ALLaVA/allava_laion/images/ | wc -l
```

### Q3: 可以中断下载后继续吗？

**A**: 可以！脚本使用 `wget -c`（继续下载），可以随时中断并重新运行。

### Q4: 下载速度太慢怎么办？

**A**:
1. 使用 HuggingFace 镜像：`export HF_ENDPOINT=https://hf-mirror.com`
2. 使用多线程下载工具（如 aria2）
3. 考虑使用云服务器（靠近 HuggingFace 服务器的地区）

### Q5: 部分数据训练后能评估加速效果吗？

**A**: 可以！加速效果主要取决于**模型架构**而非训练数据量。即使用少量数据训练，也能评估推理加速比（可能略低于完整训练）。

---

## 🎯 总结

| 如果您... | 推荐方案 | 预计时间 |
|---------|---------|---------|
| 只想看看流程 | 方案3（最小数据集） | 1小时 |
| 想用真实数据快速验证 | 方案1（1个图像包） | 半天 |
| 想要较好的训练效果 | 方案1（2-3个图像包） | 1-2天 |
| 想要接近完整训练 | 方案1（5+个图像包） | 3-5天 |
| 正式训练发布 | 完整数据集 | 1-2周 |

**下一步**：
1. 选择合适的方案
2. 按步骤准备数据
3. 修改训练脚本
4. 开始训练
5. 评估效果

祝训练顺利！🚀
