# KV 卸载与实例规模计算器（简化版）

单页静态计算器，无后端、数据不出浏览器。回答三个问题：

1. **可承载实例数**——选定模型与精度后，全部计算节点能跑几个实例（每实例所需节点数一并给出）；
2. **单 token KV 大小**——按模型注意力架构（GQA / MLA / 混合线性）× KV 量化精度计算；
3. **KV 卸载设备的容量与读带宽**——按请求到达率 × 单请求 KV × 保留窗口计算。

14 款模型开源模型（架构参数解析自各模型 HuggingFace `config.json` 与模型卡片）：
Qwen3.8 系列、GLM-5.x、DeepSeek-V4.1-Flash、Kimi-K3 / K3.1-Pro、MiMo-V2.6、HunYuan Hy3 / Hy4-preview、MiniMax-M3 / M3.1-Flash。

## 计算口径

独立估算链（单页无后端，全部计算在浏览器内完成）：

```
每实例节点数 = ⌈(权重 + 在途×单请求KV×KVH) ÷ (GPU数 × 显存 × 90%)⌉（KVH=0.5：仅一半在途 KV 常驻 HBM，另一半常驻卸载设备，v1.6）
                再沿节点数向上取最小合法并行网格：TP 须整除注意力头数与 hidden 维，
                PP 须整除层数（节点内 TP 优先；如 Qwen3.8-2.4T 的 92 层 ⇒ 3/5/6/7 节点不可行，4 节点 TP8×PP4 起）
实例数       = ⌊节点总数 ÷ 每实例节点数⌋
单请求 KV    = 上下文长度 × 单token KV + 线性注意力固定状态（FP32）
到达率       = 实例数 × 在途 ÷ (prefill时长 + 输出长÷tps)
写带宽       = 到达率 × 单请求KV          读带宽 = 写带宽 × 1.0
容量(主KPI) = 写带宽 × 保留窗口 × 1.2（保留窗口制口径）
容量(工作集)= 实例数 × 在途 × 单请求KV × 1.2（在途工作集口径，明细并列）
```


固定假设（页脚有完整释义）：MFU 50%、显存利用率 90%、冗余 1.2、读放大 1.0、去重率 0；DSA 稀疏注意力索引状态按 KV 同精度计入单 token KV。
「单 GPU 算力」填**矩阵乘实际执行精度的稠密（非稀疏）峰值**：BF16/FP16 权重按 BF16 档、FP8 权重按 FP8 档、
INT4（weight-only 量化，实际以 BF16 算力执行）按 BF16 档；默认 2.0 PF = H100/H200 的 FP8 稠密峰值（1979 TFLOPS）。
PD 分离按权重双副本建模。

**TP 复制**：TP 超过 KV 头数时 KV 在卡间复制（GQA ⌈TP/KV头⌉ 份，MLA 按 TP 份整份复制，
vLLM/SGLang 标准行为），显存需求与水位按复制计，卸载池仍存单份。

**校核（v1.1 拆分）**：单请求 prefill vs TTFT 目标；prefill 算力占用（到达率×时长）与
decode 吞吐占用（÷参考聚合 dec_agg=350 tok/s/卡×实例GPU数×算力缩放）分列，并给出"算力可持续在途"；
KV 水位阈值 70%（工程经验阈值）。

## 官方缓存语义与精度默认（v1.3 / v1.4，重要）
- **DeepSeek-V4.1-1017**（官方 `inference/model.py`）：KV 仅 4 个源层拥有且**池化压缩**（compress_ratio 2/2/2/1 → 每 token 2.5 条 512 维 latent，rope 内含于 512）；indexer 为 4 份源层共享的 128 维单键（32 个 index 头是查询侧概念，不复制键）；每层 128×512 FP8 滑窗环（含 3 DSpark draft）按每序列常量计。合计 1,600 元素/token + 2.82M/序列（v1.3 前误按 35,072 元素/token，高估 21.9×）。
- **DeepSeek-V4-Flash-0731**：21 层 ratio=4 + 20 层 ratio=128 交错池化（5.41 条×512/token + 5.25 条×128 indexer），46 层滑窗。
- **Kimi-K3 / K3.1-Pro**：69 KDA 线性层 + 24 MLA 全注意力层混合；MLA 576/token；KDA 循环态 116.2M 元素/序列（FP32）。
- 对标 [kvcache.ai KV Cache Calculator](https://kvcache.ai/tools/kv-cache-size-calculator/)：其 V4.1 "890 B/token @8K" 为官方 FP4 载荷含量化 scale；本工具 FP4 档 = 800 B（不含 scale）+ 滑窗摊销，结构口径一致。
- **精度默认随模型**：切换模型时权重精度置为原生发布精度（safetensors dtype 桶 + quantization_config 判定，`detect_wdt`）；KV 默认：FP8/FP4 权重模型 → FP8，BF16 模型 → BF16。FP8 注意力+FP4 专家的混合检查点在 FP4 档按 0.5 B/参数统计，低估注意力部分 10~25%。
- **dec_agg**：decode 聚合吞吐参考 = 350 tok/s/卡 × 实例 GPU 数 × (算力÷2.0)（"32 卡×350=11,200 tok/s" 外推假设，务必实测替换）。
- **v1.5 全模型对标 kvcache.ai**：DSA/MSA 索引键统一为 **MQA 单键 × 拥有层**（GLM-5.2/5.3、Hy4 → 21×128=2,688/token；MiniMax-M3 → 57×128=7,296；M3.1 → 60×128=7,680；Qwen3.8-Flash-Next → 12×128÷4=384，估计值）；规则以 DeepSeek-V3.2 实测口径验证（61 层全索引 → 42,944 元素/token）。MiMo-V2.6 = **9(10) 全层 + 39(60) 滑窗层**（窗口 8头×(192+128)×128 每序列快照，随 KV 精度）。GLM/M3 两家精确一致；V4 系差异仅为 draft 层窗口开关。

## 使用

打开 https://fred-chen.github.io/kv-ratio-calculator/ 即可，无需安装。
