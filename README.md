# 🎯 戒毒知识库RAG问答系统

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![vLLM](https://img.shields.io/badge/vLLM-0.6.5-green.svg)](https://github.com/vllm-project/vllm)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**基于 Qwen2.5-7B + BGE-M3 的高性能政策咨询系统**

[快速开始](#-快速开始) • [技术架构](#-技术架构) • [性能数据](#-性能数据) • [项目亮点](#-项目亮点)

</div>

---

## 📖 项目简介

本项目是一个**生产级RAG问答系统**，针对戒毒政策咨询场景设计，通过检索增强生成（RAG）技术提供准确、专业的政策咨询服务。

### 🌟 核心特性

- 🚀 **高性能**: 10并发 QPS 2.86，TTFT均值 0.27s（P99 < 0.94s），已在50人业务场景落地验证
- 🎯 **高质量**: 96%准确率，经人工深度评估验证
- 🔍 **多策略检索**: 向量检索、混合检索、Reranker重排序
- 📊 **完整测试体系**: 性能压测 + 质量评估 + A/B对比实验
- 🛡️ **生产就绪**: 完善的监控、日志、错误处理机制

---

## 🏗️ 技术架构

### 系统流程
```
用户问题
   ↓
Query预处理 & 意图识别
   ↓
向量检索 (BGE-M3 Embedding)
├─ Top-K: 10
└─ Cosine Similarity
   ↓
可选: Reranker重排序
├─ BGE-Reranker-v2-m3
└─ Top-N: 5
   ↓
Context构建 & Prompt注入
   ↓
LLM生成 (Qwen2.5-7B + vLLM)
├─ Temperature: 0.1
└─ Max Tokens: 512
   ↓
结构化答案
```

### 技术栈

| 模块 | 技术选型 | 版本 | 说明 |
|-----|---------|------|------|
| **LLM** | Qwen2.5-7B-Instruct | - | 阿里千问，中文SOTA |
| **推理引擎** | vLLM | 0.6.5 | 高性能推理，PagedAttention |
| **向量模型** | BGE-M3 | - | BAAI出品，1024维 |
| **Reranker** | BGE-Reranker-v2-m3 | - | 精排模型（可选） |
| **向量数据库** | ChromaDB | 0.4.18 | 轻量级向量库 |
| **框架** | LangChain | 0.1.0 | RAG编排框架 |
| **硬件** | NVIDIA A6000 | 48GB | GPU加速 |

---

## 📊 性能数据

### 压测结果（10并发，每方法30请求，max_tokens=512）

| 指标 | Baseline | Rerank | Hybrid+Rerank |
|-----|----------|--------|---------------|
| **QPS** | 2.86 | 2.46 | 2.66 |
| **TTFT均值** | 0.27s | 0.46s | 0.35s |
| **TTFT P99** | 0.94s | 1.99s | 1.67s |
| **总响应P99** | 4.67s | 4.97s | 5.17s |
| **成功率** | 100% | 100% | 100% |

### 质量评估（人工深度评估，5维度）

| 方法 | 准确率 | 完整性 | 流畅度 | 综合得分 | 排名 |
|-----|--------|--------|--------|---------|------|
| **Baseline** | **96%** ✅ | 95% | 98% | **4.80/5** | 🥇 |
| Rerank | 82% | 88% | 94% | 4.10/5 | 🥈 |
| Hybrid+Rerank | 57% ❌ | 65% | 92% | 2.85/5 | 🥉 |

**关键发现**: 在高质量法律文档场景下，纯向量检索（Baseline）优于复杂混合方案。

---

## 🎯 项目亮点

### 1. 发现"反向优化"现象 

**现象**: Hybrid+Rerank方案质量反而下降39%（96% → 57%）

**深度分析**:

1. **检索偏离**: Reranker在"戒毒人员权利"问题上答案完全跑偏，回答了"程序性权利"（外出探视、会见律师）而非"基本权利"（不受歧视、信息保密）

2. **信息丢失**: "强制隔离戒毒定义"问题缺少"例外情况"关键信息

3. **知识库特性不匹配**: 高质量单一来源法律文档，向量检索已接近最优

**工程启示**: 
- 技术选型要看数据特性，不是越复杂越好
- 遵循"奥卡姆剃刀"原则：简单方案更优
- 实测胜于理论，避免过度工程化

---

### 2. 完整的工程化实践 

**性能测试体系**
- 10并发压测，验证系统稳定性
- 多维度监控（QPS、延迟、TTFT、成功率）
- 自动化测试脚本，可重复执行

**科学的质量评估**
- 语义相似度自动评估（BGE-M3 Embedding）
- 人工深度评估（准确性、完整性、流畅度等5维度）
- A/B对比实验，验证不同策略效果

**生产级代码规范**
- 模块化设计，高内聚低耦合
- 完整的日志系统（loguru）
- 错误处理与降级策略
- 详细的代码注释和文档

---

### 3. vLLM性能优化 

| 优化项 | 优化前 | 优化后 | 提升 |
|--------|--------|--------|------|
| **GPU利用率** | 68% | 92% | +35% |
| **QPS** | 0.45 | 0.77 | **+71%** |
| **TTFT** | 1.2秒 | 0.9秒 | -25% |
| **检索延迟** | 85ms | 35ms | -59% |

**关键配置**:

- tensor_parallel_size: 1（单卡部署）
- max_num_seqs: 16（并发批处理）
- gpu_memory_utilization: 0.95（最大化显存利用）
- PagedAttention: 减少显存碎片

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.4+
- GPU: A6000
- 系统内存: 64GB+
- 硬盘空间: 50GB+（模型文件）

### 安装

克隆项目
```
git clone https://github.com/Nobody-ly/drug-rehab-rag.git
cd drug-rehab-rag
```
安装依赖（推荐使用uv，速度快10倍）
```
pip install uv
uv sync
```
或使用传统方式
```
pip install -r requirements.txt
```
### 下载模型

方式1: 使用huggingface-cli（国外）
```
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/Qwen2.5-7B-Instruct
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir models/bge-reranker-v2-m3
```
方式2: 使用ModelScope（国内更快）
```
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir models/Qwen2.5-7B-Instruct
modelscope download --model BAAI/bge-m3 --local_dir models/bge-m3
```
### 运行系统

1. 启动vLLM服务
```
bash scripts/start_vllm.sh
```
等待模型加载完成，看到"Application startup complete"

2. 构建向量数据库（首次运行）
```
python core/build_vectordb.py
```
预计耗时: 2-5分钟（取决于文档数量）

3. 运行单次问答
```
python core/rag_system.py --query "什么是强制隔离戒毒？"
```
4. 运行完整压测
```
bash scripts/run_benchmark.sh
```
压测时间约: 5-10分钟

---

## 📁 项目结构
```
drug-rehab-rag/
├── core/                           # 核心代码
│   ├── rag_system.py              # RAG主系统
│   ├── build_vectordb.py          # 向量库构建
│   ├── semantic_quality_eval.py   # 自动质量评估
│   ├── quality_evaluation.py      # 人工质量评估
│   └── document_loader.py         # 文档加载器
│
├── benchmark/                      # 压测代码
│   ├── full_benchmark.py          # 完整压测（3种策略）
│   ├── vllm_optimization.py       # vLLM优化测试
│   ├── load_test.py               # 基础负载测试
│   └── measure_ttft.py            # TTFT测试
│
├── data/                           # 数据文件
│   ├── questions/                 # 问题集 & 标准答案
│   │   └── jieduo_qa.json
│   └── documents/                 # 知识库文档
│       ├── jieduo_documents/      # 戒毒政策文档
│       └── pdfs/                  # 原始PDF
│
├── results/                        # 测试结果
│   ├── benchmark/                 # 压测报告
│   │   ├── latest_report.json
│   │   └── latest_results.json
│   ├── optimization/              # 优化报告
│   └── evaluation/                # 质量评估
│
├── docs/                           # 文档
│   ├── FINAL_BENCHMARK_REPORT.md  # 完整测试报告
│   └── performance_baseline.md    # 性能基线
│
├── scripts/                        # 脚本
│   ├── start_vllm.sh             # 启动vLLM服务
│   └── run_benchmark.sh          # 运行压测
│
├── requirements.txt               # 依赖清单
├── pyproject.toml                # uv配置
├── README.md                     # 项目说明
└── .gitignore                    # Git忽略文件
```
---

## 🧪 测试

### 运行性能压测

完整压测（3种检索策略对比）
```
python benchmark/full_benchmark.py
```
查看结果
```
cat results/benchmark/latest_report.json
```
### 运行质量评估

自动评估（语义相似度）
```
python core/semantic_quality_eval.py results/benchmark/latest_results.json
```
人工评估
```
python core/quality_evaluation.py results/benchmark/latest_results.json
```
### 单元测试
```
pytest tests/ -v
```
---

## 📈 后续优化方向

### 短期（1-2周）

**1. Prompt优化** (优先级: 高)
- 增加Few-shot示例，提升答案质量
- 引入Chain-of-Thought，增强推理能力
- 自适应长度控制，避免过长或过短

**2. Query改写** (优先级: 中)
- 同义词扩展，提升召回率
- 法律术语规范化，统一表述
- 意图识别，区分不同类型问题

### 中期（1个月）

**3. Self-RAG** (优先级: 中)
- 检索结果可靠性判断
- 动态调整检索策略
- 答案自验证机制

**4. 缓存机制** (优先级: 高)
- 常见问题缓存，减少重复计算
- Embedding缓存，加速检索
- 答案缓存，提升响应速度

### 长期（3个月）

**5. 多模态支持** (优先级: 低)
- PDF文档直接解析（无需预处理）
- 图表信息提取
- OCR识别手写文档

**6. 分布式部署** (优先级: 低)
- 多GPU并行推理
- 负载均衡
- 服务高可用

---

## 📝 相关文档

- [完整压测报告](docs/FINAL_BENCHMARK_REPORT.md) - 详细的性能和质量评估
- [性能基线](docs/performance_baseline.md) - 系统性能基准
- [技术架构设计](docs/architecture.md) - 待补充
- [vLLM优化手册](docs/vllm_optimization.md) - 待补充
- [Prompt工程指南](docs/prompt_engineering.md) - 待补充

---

## 🔧 常见问题

### Q1: vLLM服务启动失败？

**原因**: GPU显存不足或CUDA版本不匹配

**解决方案**:
- 检查GPU显存: nvidia-smi
- 降低gpu_memory_utilization: 0.95 → 0.85
- 检查CUDA版本: nvcc --version

### Q2: 向量数据库构建太慢？

**原因**: 文档数量太多或embedding模型慢

**解决方案**:
- 使用GPU加速: device='cuda'
- 减少文档数量（测试时）
- 使用更小的embedding模型

### Q3: 答案质量不理想？

**原因**: 检索不准确或prompt不合适

**解决方案**:
- 调整Top-K参数（增大到15-20）
- 优化Prompt模板
- 使用Reranker精排（如果baseline不够好）
- 增加Few-shot示例

### Q4: QPS无法满足需求？

**原因**: 单机性能瓶颈

**解决方案**:
- 使用多GPU并行（tensor_parallel_size=2）
- 增加max_num_seqs（批处理大小）
- 部署多个服务实例，负载均衡
- 使用更小的模型（如Qwen2.5-3B）

---

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能LLM推理引擎
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - 阿里千问开源大模型
- [BGE-M3](https://github.com/FlagOpen/FlagEmbedding) - BAAI中文向量模型
- [LangChain](https://github.com/langchain-ai/langchain) - RAG框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 向量数据库

---

## 📄 License

本项目采用 [MIT License](LICENSE)

---

## 👤 作者

**Nobody-ly**

- GitHub: [@Nobody-ly](https://github.com/Nobody-ly)
- Email: 2811925029@qq.com
- 学校: 西安电子科技大学
- 专业: 计算机技术

---

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 提交 [Issue](https://github.com/Nobody-ly/drug-rehab-rag/issues)
- 发送邮件至: 2811925029@qq.com

---

<div align="center">

**Made with ❤️ by Nobody-ly**

[⬆ 回到顶部](#-戒毒知识库rag问答系统)

</div>
