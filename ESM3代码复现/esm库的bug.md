# ESM3代码复现

## 调用本地模型

```python
import torch
from esm.models.esm3 import ESM3
from esm.tokenization import get_esm3_model_tokenizers

# 设备 & 模型（统一用 float32 避免 BFloat16 兼容问题）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM3.from_pretrained("esm3_sm_open_v1").eval().float().to(device)
tokenizer = get_esm3_model_tokenizers().sequence

# 输入序列
sequences = ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"]
inputs = tokenizer(sequences, return_tensors="pt")
sequence_tokens = inputs["input_ids"].to(device)

# 推理
with torch.inference_mode():
    output = model(sequence_tokens=sequence_tokens)

# === 推理结果 ===

# 1. Embeddings
print(f"Embeddings 形状: {output.embeddings.shape}")  # [B, L, 1536]

# 2. 预测氨基酸序列（logits → argmax → decode）
pred_ids = output.sequence_logits.argmax(dim=-1)         # [1, L]
pred_seq = tokenizer.decode(pred_ids[0].tolist())         # 去掉 BOS/EOS/特殊 token
print(f"\n原始序列:\n{sequences[0]}")
print(f"\n预测序列:\n{pred_seq}")

# 3. 每个位置的氨基酸概率
probs = output.sequence_logits.softmax(dim=-1)            # [1, L, 33]
max_prob, max_id = probs.max(dim=-1)                      # 每个位置最大概率
print(f"\n平均置信度: {max_prob[0, 1:-1].mean().item():.4f}")  # 去
```

### 1. 为什么 `model.float().to(device)` 能解决 dtype 不匹配？

首先，矩阵乘法（`matmul`）要求两个操作数的 dtype **必须严格一致**（除非使用特定的自动混合精度上下文）。

- **冲突现场**：模型权重是 `BFloat16`，但硬编码的 `average_plddt` 计算中生成了 `Float32` 的张量。当 `BFloat16` 矩阵与 `Float32` 矩阵相乘时，PyTorch 会直接抛出 `RuntimeError: expected scalar type BFloat16 but found Float`。
- **方案逻辑**：`model.float()` 会将模型**所有的参数（weights）和缓冲区（buffers）** 强制转换为 `Float32`。此时，模型权重变成了 Float32，与 `average_plddt` 输出的 Float32 完全一致，矩阵乘法顺利通过。这相当于**用显存空间换取了代码运行的绝对兼容性**，绕过了复杂的类型判断逻辑。

### 2. 为什么要强调 `float().to(device)` 的顺序？（关键）

先转设备再转浮点（`to(device).float()`）不行吗？**绝对不行，这恰恰是解决显存溢出（OOM）的精髓所在。**

- **`float().to(device)`（先转后移）**：在 **CPU 内存**中完成 BFloat16 -> Float32 的转换，然后将转换好的 Float32 模型拷贝到 GPU 显存。**优势**：转换过程不占用宝贵的 GPU 显存，且 CPU 内存通常远大于 GPU 显存，几乎不会爆内存。
- **`to(device).float()`（先移后转）**：先将巨大的 BFloat16 模型加载到 GPU 显存，然后在 GPU 上原地转换为 Float32。**致命缺陷**：在转换瞬间，PyTorch 需要在 GPU 显存中**同时保留** BFloat16 的旧副本和 Float32 的新副本，导致显存占用瞬间翻倍（近乎 2x 模型大小），极易触发 CUDA Out of Memory。

所以，这个顺序并非为了逻辑正确，而是为了**在不爆显存的前提下，强行抹平所有数据类型差异**。

### 补充：BFloat16 到底是什么？

**BFloat16**（Brain Floating Point Format）是谷歌提出的一种浮点数格式，专为AI计算设计。

- **内存布局（1-8-7）**：1位符号位 + **8位指数位** + 7位小数位。
- **与 Float32 对比**：Float32 是 1-8-23；BFloat16 直接**截断了 Float32 的后16位小数**，但**完整保留了 Float32 的 8 位指数范围**。
- **与 Float16 对比**：Float16 是 1-5-10，指数范围窄，容易在训练中大数溢出（上溢）或极小梯度下溢出（下溢）。而 BFloat16 的动态范围与 Float32 **完全一致**，因此在 ESM3 这种深层模型中，它既能将显存占用减半、加速计算，又不会像 Float16 那样频繁出现数值溢出。

**它的缺点**：精度（小数位数）较低，但对于蛋白质结构预测这种对“相对大小”比对“绝对小数”更敏感的任务，影响微乎其微。

### 总结建议

使用 `model.float().to(device)` 是**用 2 倍显存占用（相比 BF16）换取 100% 兼容性和代码稳定性**的做法。如果你的显卡显存足够大（如 80GB A100），这是最优解；如果显存紧张，根治方案应该是**修改 `average_plddt` 源码**，在其 `.float()` 后添加 `.to(dtype=model.dtype)`，让输出随模型权重动态适配，从而继续享受 BFloat16 的低显存红利。

#### 能够享受BFloat的轻量显存

```python
import torch
from esm.models.esm3 import ESM3
from esm.tokenization import get_esm3_model_tokenizers

# 清空显存缓存（防止上次运行残留）
torch.cuda.empty_cache()

# 设备 & 模型（GPU 上使用 BFloat16 节省显存，CPU 回退 float32）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ESM3.from_pretrained("esm3_sm_open_v1").eval()
if device.type == "cpu":
    model = model.float().to(device)
else:
    model = model.to(device)  # BFloat16，源码已修复 dtype 匹配

print(f"模型 dtype: {next(model.parameters()).dtype}")

tokenizer = get_esm3_model_tokenizers().sequence

# 输入序列
sequences = ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"]
inputs = tokenizer(sequences, return_tensors="pt")
sequence_tokens = inputs["input_ids"].to(device)

# 推理
with torch.inference_mode():
    output = model(sequence_tokens=sequence_tokens)

# 1. Embeddings
print(f"Embeddings 形状: {output.embeddings.shape}")

# 2. 预测氨基酸序列
pred_ids = output.sequence_logits.argmax(dim=-1)
pred_seq = tokenizer.decode(pred_ids[0].tolist())
print(f"\n原始序列:\n{sequences[0]}")
print(f"\n预测序列:\n{pred_seq}")

# 3. 置信度
probs = output.sequence_logits.softmax(dim=-1)
max_prob, _ = probs.max(dim=-1)
print(f"\n平均置信度: {max_prob[0, 1:-1].mean().item():.4f}")
```

这里相当不好改，让人不禁怀疑ESM3这个库的开发者是不是故意瞎搞的。

`plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt).to(dtype=self.dtype)) `这部分需要 `self.dtype` 利用 `self` 让 `dtype` 和模型的参数类型保持一致。

| 行号    | 位置                   | 改前                       | 改后                                            |
| ------- | ---------------------- | -------------------------- | ----------------------------------------------- |
| 116-118 | `EncodeInputs.forward` | `rbf_16_fn(average_plddt)` | `rbf_16_fn(average_plddt).to(dtype=self.dtype)` |
| 332-333 | `ESM3.forward`         | `defaults(...).float()`    | `defaults(...).to(dtype=self.dtype)`            |

**原理：** `self` 在两个类里都是 `nn.Module`，`self.dtype` 自动取模型参数的当前 dtype。GPU 上 `from_pretrained` 把模型转成 BFloat16 后，所有层参数都是 BFloat16，`self.dtype` 自然就是 `torch.bfloat16`，不会出现 Float32 张量进 BFloat16 层的 dtype 不匹配。

然后又报错了。

`plddt_embed = self.plddt_projection(rbf_16_fn(average_plddt).to(dtype=next(self.parameters()).dtype))` 这才是正确的改法

#### 1. `next(self.parameters()).dtype` 是什么

```python
self.parameters()   # → 一个生成器，逐个 yield 模型的所有参数张量
next(...)            # → 取出第一个参数张量（比如 Embedding 的 weight）
.dtype               # → 该张量的 dtype，如 torch.bfloat16
```

PyTorch 的 `nn.Module` 用**统一 dtype 管理参数**——`from_pretrained` 里这行保证了所有参数是同一 dtype：

```python
# esm3.py line 240-241
model = load_local_model(model_name, device=device)
if device.type != "cpu":
    model = model.to(torch.bfloat16)   # 所有参数一起转为 bfloat16
```

所以取任意一个参数的 `.dtype` 就能代表整个模型的当前 dtype，不需要 `ESM3` 额外定义 `.dtype` 属性。

------

#### 2. 为什么 `EncodeInputs` 里也能用 `self`

```python
class EncodeInputs(nn.Module):   # 它也是 nn.Module 的子类
    def __init__(self, d_model):
        self.sequence_embed = nn.Embedding(...)   # 有参数
        self.plddt_projection = nn.Linear(...)     # 有参数
```

`EncodeInputs` 作为 ESM3 的**子模块**（被 `self.encoder = EncodeInputs(...)` 持有），当父模型 `model.to(torch.bfloat16)` 时，PyTorch 会递归地把所有子模块的参数一起转。所以 `EncodeInputs.parameters()` 取出的参数也全是 BFloat16。



一句话总结：**PyTorch 的参数 dtype 是全局一致的，取任意一个参数就能拿到全模型的 dtype，用它在关键入口做对齐。**

---

## 关于esm这个库的bug

### 总结：所有改动

共修改 **2 个文件**，修复了 **5 个 dtype 断裂点**：

#### 文件 1：`esm\models\esm3.py`（4 处）

| 行号    | 改什么                                                       | 原因                                            |
| ------- | ------------------------------------------------------------ | ----------------------------------------------- |
| 332-333 | `defaults(x).float()` → `.to(dtype=next(self.parameters()).dtype)` | 默认 plddt 张量不能硬写 Float32                 |
| 348     | `dtype=torch.float` → `dtype=next(self.parameters()).dtype)` | structure_coords 填空值需匹配模型               |
| 116-118 | `rbf_16_fn(x)` → `rbf_16_fn(x).to(...)`                      | RBF 内 `torch.exp` 对 BFloat16 可能回退 Float32 |
| 355     | 新增 `affine = affine.to(...)`                               | 坐标运算后转回模型 dtype（trans 部分有效）      |

#### 文件 2：`esm\layers\geom_attention.py`（2 处）

| 行号 | 改什么                                                       | 原因                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 56   | `.float()` → `.to(dtype=s.dtype)`                            | attn_bias 需匹配输入 dtype                                   |
| 147  | `out_proj(attn_out)` → `out_proj(attn_out.to(dtype=s.dtype))` | **关键**——`RotationMatrix.__init__` 强制 Float32，几何注意力全程 Float32，进入 `out_proj` 前必须转回 |

根本原因是你最初问的——`RotationMatrix` 构造函数里第 90 行 `self._rots = rots.to(torch.float32)` 死锁，不管外面怎么 `.to(dtype=...)` 都给你转回 Float32。这是库的设计取舍（旋转矩阵需要精度），所以最终防御线在 `out_proj` 之前。



我问Claude Code：既然几何注意力必然会出现float32，那为什么微软还要开发一个BFloat16用于ESM3？这部纯恶心人吗？

回答：不是恶心人，是他们的 BFloat16 支持**没写完整**——这是个 bug，不是设计决策。

#### BFloat16 的设计意图是正确的

ESM3 有 28 层 transformer，其中**只有 1 层**用了几何注意力（`n_layers_geom=1`）：

```
Layer 0:  Geometric Attention  ← 这 1 层需要 Float32 旋转精度
Layer 1-27: Standard MHA + FFN  ← 这 27 层全可以跑 BFloat16
```

所以 BFloat16 确实能省**绝大多数**的显存和计算——只是他们写完 `from_pretrained` 把模型转了 BFloat16 之后，**忘了把 forward 里硬编码 Float32 的 5 处也改掉**。

#### 根因就这一行

```python
# affine3d.py:90 — RotationMatrix.__init__
self._rots = rots.to(torch.float32)  # 死锁：无论传什么 dtype，强制转 Float32
```

这是**故意的**——旋转矩阵的 Gram-Schmidt 正交化、`torch.einsum` 这些运算需要 Float32 精度，BFloat16 尾数只有 7 位，在 3D 坐标变换里会累积很大误差。

**正确做法**是：旋转计算内部用 Float32 保证精度，但**输出前转回模型 dtype**，就像我们最后在 `out_proj` 前加的那行：

```python
s = self.out_proj(attn_out.to(dtype=s.dtype))  # ← 应该由库自己做
```

#### 为什么他们没发现？

这个 bug 在以下条件会**被掩盖**：

- **老 GPU 架构**（V100/P100 及以前的 FP16）：`from_pretrained` 检测到 GPU 是 `torch.float16` 而非 `torch.bfloat16`，整个问题链不触发
- **CUDA Compute Capability 8.0+**（A100、RTX 4090 等）：某些架构下 `torch.exp` 有 BFloat16 原生 kernel，RBF 不回退 Float32，断裂点③不触发
- 断裂点①②④⑤只在同时满足「Ampere+ 架构 + 特定 CUDA 版本 + 驱动」的组合下才会炸

说白了这库主要在 A100/H100 集群上测试，你用的 RTX 5060 有些 kernel 路径不一样，才踩到了坑。

#### 使用BFloat16之后

| 指标           | 数值                               |
| -------------- | ---------------------------------- |
| 模型 dtype     | `torch.bfloat16`                   |
| 参数量         | 14.0 亿                            |
| 显存占用       | **2.67 GiB**（FP32 需要 5.35 GiB） |
| 省显存         | **2.67 GiB（50%）**                |
| 推理耗时       | **3.33s**                          |
| 所有输出 dtype | 全部 BFloat16 ✓                    |

你的 RTX 5060（8GB）跑 BF16 绰绰有余，如果用 FP32 的话 OOM 边缘。现在 notebook 里那行也不用 `.float()` 了，直接享受省下来的 2.67GB 显存。