# ESMC 数据集批处理方案

## 背景

用户希望用 ESMC-600M 处理多条蛋白质序列（数据集），而非单个 GFP 序列。需要理解 ESMC 如何处理变长序列批次，并添加相应的 Cell 代码。

## ESMC 批处理原理

### 输入流程

```
["MKS...", "MAAG...", "MSDE..."]  (3条长度不同的序列)
        │
        ▼
每条单独 tokenize: encode(seq) → [cls, tokens..., eos]
        │
        ▼
stack_variable_length_tensors() → 补 pad 到 max_len
        │
        ▼
sequence_tokens: (B, L_max) = (3, 142)
sequence_id:     (B, L_max) = 布尔 mask (True=真实token, False=pad)
        │
        ▼
Flash Attention: unpad_input → 拼接所有真实token为1D → 计算注意力 → pad_input 恢复 (B, L, D)
```

### 关键 API

- **Tokenizer**: `tokenizer.encode(seq)` 返回 token ID list（含 cls/eos）
- **Padding**: `esm.utils.misc.stack_variable_length_tensors(tensors, constant_value=pad_id)`
- **sequence_id**: `sequence_tokens != tokenizer.pad_token_id` 自动生成布尔 mask
- **Forward**: `model(sequence_tokens=seq, sequence_id=mask)` 返回 `ESMCOutput`

### ESMCOutput 字段

| 字段 | 形状 | 含义 |
|------|------|------|
| `embeddings` | `(B, L_max, d_model)` | 每位置嵌入 |
| `hidden_states` | `(n_layers, B, L_max, d_model)` | 所有层隐状态 |
| `sequence_logits` | `(B, L_max, 64)` | 氨基酸预测 logits |

## 实现方案

新增 **Cell 5** — ESMC 数据集批处理：

### Step 1: 构建示例数据集
```python
# 6 条来自不同家族的蛋白质，长度各异
dataset = {
    "GFP":       "MSKGEELFTGVVPILVELDGDVNGHK...",  # 238 aa - GFP
    "Lysozyme":  "KVFGRCELAAAMKRHGLDNYRGYSL...",  # 129 aa - 溶菌酶
    "Ubiquitin": "MQIFVKTLTGKTITLEVEPSDTIENV...",  # 76 aa  - 泛素
    "Myoglobin": "MGLSDGEWQLVLNVWGKVEADIPGHG...",  # 154 aa - 肌红蛋白
    "Insulin":   "MALWMRLLPLLALLALWGPDPAAAFV...",  # 110 aa - 胰岛素
    "RNase A":   "KETAAAKFERQHMDSSTSAASSSNYC...",  # 124 aa - 核糖核酸酶
}
```

### Step 2: 手动批处理（展示底层原理）
```python
from esm.utils.misc import stack_variable_length_tensors

# 逐条 tokenize
all_tokens = []
for name, seq in dataset.items():
    ids = tokenizer.encode(seq, add_special_tokens=True)  # [cls, ...aa..., eos]
    all_tokens.append(torch.tensor(ids))

# 堆叠 + padding
batch_tokens = stack_variable_length_tensors(all_tokens, constant_value=tokenizer.pad_token_id)
batch_mask = batch_tokens != tokenizer.pad_token_id  # (B, L_max)

# 推理
with torch.inference_mode():
    batch_output = esmc600_model(sequence_tokens=batch_tokens, sequence_id=batch_mask)
```

### Step 3: 逐序列分析

```python
# 每序列的嵌入（去除 padding 和 BOS/EOS）
for i, (name, seq) in enumerate(dataset.items()):
    valid_len = (batch_mask[i] == True).sum().item()  # 含 cls+eos
    emb = batch_output.embeddings[i, 1:valid_len-1, :]  # (L_i, d_model)
    
    # 预测序列
    pred_ids = batch_output.sequence_logits[i, 1:valid_len-1, :].argmax(dim=-1)
    pred_seq = tokenizer.decode(pred_ids.tolist())

    # 准确率
    correct = sum(1 for a, b in zip(seq, pred_seq.replace(" ", "")) if a == b)
    print(f"{name}: 长度={len(seq)}, 准确率={correct/len(seq)*100:.1f}%")
```

### Step 4: 跨序列分析
- **Embedding 相似度矩阵**: 各蛋白在嵌入空间的 pairwise cosine similarity
- **Perplexity 对比**: 不同蛋白家族的建模难度
- **层间表示演化**: 提取 early/mid/late 层 hidden_states，观察特征如何逐层抽象

## 关键技术细节

### 重要：ESMC 的跨序列注意力

ESMC 的 `sequence_id` 只是 padding mask（True/False），**不是** per-sequence ID。这意味着批次中所有序列的真实 token 会**互相注意**——序列 A 的 token 可以 attend 到序列 B。这在计算 per-residue embedding 时会引入跨序列"污染"。

**影响评估**：
- 如果只用来提取 `embeddings` 或 `sequence_logits`：影响很小，因为 output head 是 per-position 的
- 如果依赖 self-attention 的内部激活模式（如 SAE）：建议单条推理

如果需要严格隔离，应该逐条推理：
```python
for i, tokens in enumerate(all_tokens):
    single = tokens.unsqueeze(0)  # (1, L)
    mask = single != tokenizer.pad_token_id
    out = model(sequence_tokens=single, sequence_id=mask)
```

### HuggingFace tokenizer 可直接处理批次

如果不想手动 padding，也可以用 HF 标准接口：
```python
encoded = tokenizer(list(dataset.values()), padding=True, return_tensors="pt")
# encoded["input_ids"] → (B, L_max)，自动 padded
# encoded["attention_mask"] → (B, L_max)，可当 sequence_id 用
```

但需确认 `EsmSequenceTokenizer` 的 `pad_token_id` 已正确定义（它是 2，查看过源码确认）。

## 涉及文件

- 仅修改 `d:\PythonProject\论文复现\ESM3.ipynb`，新增 Cell 5
- 依赖 Cell 4 已加载的 `esmc600_model` 和 `esmc600_tokenizer`

## 验证方式

运行 Cell 4（加载 ESMC-600M）→ 运行 Cell 5（批处理）→ 检查：
1. 6 条序列全部成功推理
2. 每条预测序列长度与输入一致
3. 准确率、嵌入维度、跨序列相似度正常输出
