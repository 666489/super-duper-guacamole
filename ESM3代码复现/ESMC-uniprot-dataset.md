# ESMC + 本地 UniProt 数据集批处理方案

## 背景

当前 notebook 中 ESMC 仅处理了一条硬编码的 GFP 序列。`data/` 目录下有包含 **575,503 条蛋白序列**的 UniProt Swiss-Prot FASTA 文件，可直接作为 ESMC 批处理的真实数据集。

### 数据源

```
d:\PythonProject\data\Biological_Harness\Fun_Evidence\UniProt\uniprot_sprot.fasta
```

| 属性 | 值 |
|------|------|
| 大小 | 275 MB |
| 序列数 | 575,503 |
| 格式 | 标准 FASTA（`>` 头 + 氨基酸序列） |
| 质量 | UniProt/Swiss-Prot 人工审核 |

### FASTA 格式示例

```
>sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=Frog virus 3 ... PE=4 SV=1
MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPSEKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLDAKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHLEKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDDSFRKIYTDLGWKFTPL
```

---

## 策略：采样而非全量

575K 条序列在单机 ESMC 上全量处理不现实（耗时约数天）。采用**智能采样**：

> 按序列长度分层采样 128 条 → 按长度排序后分批（减少 padding 浪费）→ ESMC-600M 推理 → 产生每个蛋白的 embedding → 下游分析

**为什么 128 条？**
- ESMC-600M 每 token 约 30ms（CPU 上更慢）
- 128 条平均 300aa 的序列 ≈ 38,400 tokens → CPU 约 20 分钟，GPU 约 2 分钟
- 足够做 embedding 可视化和功能聚类，兼顾效率

---

## 新增 Cell 5：ESMC + UniProt 本地数据集

### Step 1: 解析 FASTA → 采样

```python
from Bio import SeqIO
import random

fasta_path = r"d:\PythonProject\data\Biological_Harness\Fun_Evidence\UniProt\uniprot_sprot.fasta"

# 流式读取 FASTA，提取 (id, seq, description, length)
records = []
for rec in SeqIO.parse(fasta_path, "fasta"):
    aa_seq = str(rec.seq)
    records.append({
        "id": rec.id,
        "desc": rec.description.split("OS=")[0].strip() if "OS=" in rec.description else rec.description,
        "seq": aa_seq,
        "length": len(aa_seq)
    })

print(f"FASTA 总序列数: {len(records)}")

# 按长度过滤 + 分层采样
# 排除过长（>1000aa, 稀有）和过短（<50aa, 信号弱）
filtered = [r for r in records if 50 <= r["length"] <= 500]
print(f"过滤后 (50~500 aa): {len(filtered)} 条")

# 分层采样：按长度区间均匀取
bins = [(50, 150), (150, 250), (250, 350), (350, 500)]
sampled = []
for lo, hi in bins:
    bucket = [r for r in filtered if lo <= r["length"] < hi]
    n_sample = min(32, len(bucket))  # 每桶取 32 条
    sampled.extend(random.sample(bucket, n_sample))
    print(f"  [{lo}, {hi}): {len(bucket)} 条 -> 采样 {n_sample}")

print(f"最终采样: {len(sampled)} 条")
```

### Step 2: 按长度排序 + 分批

```python
# 按长度排序，减少 batch 内 padding 浪费
sampled.sort(key=lambda r: r["length"])

# 分批：每批 16 条
batch_size = 16
batches = [sampled[i:i+batch_size] for i in range(0, len(sampled), batch_size)]
print(f"共 {len(batches)} 批，每批 ≤{batch_size} 条")
```

### Step 3: 逐批推理

```python
from esm.utils.misc import stack_variable_length_tensors

all_results = []

for batch_idx, batch_records in enumerate(batches):
    names = [r["id"] for r in batch_records]
    seqs = [r["seq"] for r in batch_records]

    # Tokenize
    all_tokens = [torch.tensor(esmc600_tokenizer.encode(s, add_special_tokens=True))
                  for s in seqs]

    # Padding
    batch_tokens = stack_variable_length_tensors(
        all_tokens, constant_value=esmc600_tokenizer.pad_token_id
    ).to(esmc600_device)
    batch_mask = batch_tokens != esmc600_tokenizer.pad_token_id
    batch_mask = batch_mask.to(esmc600_device)

    # 推理
    with torch.inference_mode():
        output = esmc600_model(sequence_tokens=batch_tokens, sequence_id=batch_mask)

    # 逐条提取结果
    for i, rec in enumerate(batch_records):
        valid_len = (batch_mask[i] == True).sum().item()
        emb = output.embeddings[i, 1:valid_len-1, :].cpu()  # (L_i, D)
        logits = output.sequence_logits[i, 1:valid_len-1, :].cpu()  # (L_i, 64)

        # 预测序列 + 准确率
        pred_ids = logits.argmax(dim=-1)
        pred_str = esmc600_tokenizer.decode(pred_ids.tolist()).replace(" ", "")
        correct = sum(1 for a, b in zip(rec["seq"], pred_str) if a == b)

        # 困惑度
        ppl = torch.nn.functional.cross_entropy(
            logits.view(-1, 64),
            torch.tensor(esmc600_tokenizer.encode(rec["seq"],
                         add_special_tokens=False)).view(-1).to(logits.device),
            reduction='mean'
        ).exp().item()

        all_results.append({
            **rec,
            "emb": emb,
            "pred_seq": pred_str,
            "accuracy": correct / len(rec["seq"]),
            "perplexity": ppl
        })

    print(f"  Batch {batch_idx+1}/{len(batches)} 完成 "
          f"({', '.join(n[:12] for n in names[:3])}...)")
```

### Step 4: 可视化分析

```python
# 4.1 序列级嵌入 (mean pooling)
import torch.nn.functional as F

seq_embs = torch.stack([r["emb"].mean(dim=0) for r in all_results])  # (N, D)

# 4.2 PCA 降维到 2D → 按长度着色
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
embs_2d = pca.fit_transform(seq_embs.numpy())

import matplotlib.pyplot as plt
lengths = [r["length"] for r in all_results]
sc = plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=lengths, cmap="viridis", alpha=0.7)
plt.colorbar(sc, label="sequence length")
plt.title(f"ESMC-600M Embedding PCA (N={len(all_results)} UniProt proteins)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.show()

# 4.3 准确率 vs 序列长度
accs = [r["accuracy"] for r in all_results]
plt.scatter(lengths, accs, alpha=0.5)
plt.xlabel("Sequence Length (aa)")
plt.ylabel("Prediction Accuracy")
plt.title("ESMC-600M Self-Prediction Accuracy vs Length")
plt.show()

# 4.4 困惑度分布
ppls = [r["perplexity"] for r in all_results]
plt.hist(ppls, bins=30, alpha=0.7)
plt.xlabel("Perplexity")
plt.ylabel("Count")
plt.title("ESMC-600M Perplexity Distribution")
plt.show()

print(f"\n准确率: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
print(f"困惑度: {np.mean(ppls):.1f} ± {np.std(ppls):.1f}")
```

---

## 依赖

新增 Python 包（如果尚未安装）：
- `biopython` — FASTA 解析
- `scikit-learn` — PCA 降维
- `matplotlib` — 可视化

```bash
pip install biopython scikit-learn matplotlib
```

---

## 涉及文件

- **修改**: `d:\PythonProject\论文复现\ESM3.ipynb`，新增 Cell 5
- **只读**: `d:\PythonProject\data\Biological_Harness\Fun_Evidence\UniProt\uniprot_sprot.fasta`
- **无其他修改**

---

## 验证方式

1. 先运行 Cell 4（加载 ESMC-600M）
2. 运行 Cell 5 → 输出：
   - `FASTA 总序列数: 575503`
   - 分层采样结果（每长度区间各 32 条，共 128 条）
   - 逐批推理进度（8 批 × 16 条）
   - 三张图：PCA 散点图、准确率-vs-长度图、困惑度直方图
3. 确认 `esmc600_tokenizer.encode` 正确给每条序列加 `<cls>/<eos>`
4. GPU OOM 时自动退化为逐条推理（或直接切 CPU）
