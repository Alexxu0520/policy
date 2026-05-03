# Policy RAG Audit System — Qwen

中国企业合规文档审核系统。上传或粘贴文档，系统自动检索相关合规政策并由 Qwen 生成结构化审核报告。

## 系统架构

```text
Policy JSON
    └─ 中文分句切块 (sentence-aware chunking)
         ├─ BGE 向量编码 → ChromaDB (dense index)
         └─ jieba 分词    → BM25Okapi  (sparse index)

用户文档输入
    └─ Query Expansion (Qwen 生成 3 个子查询)
         ├─ Dense 检索  (BGE)  ─┐
         └─ BM25 检索          ─┴─ RRF 融合 → Top-10 候选
                                       └─ BGE Reranker
                                             └─ LambdaMART 精排
                                                   └─ Top-5 Policy 上下文
                                                         └─ Qwen2.5-3B 生成审核报告
```

## 技术栈

| 模块 | 组件 |
|------|------|
| 向量嵌入 | `BAAI/bge-large-zh-v1.5` (中文原生) |
| 稀疏检索 | BM25Okapi (`rank_bm25`) + jieba 分词 |
| 检索融合 | Reciprocal Rank Fusion (RRF) |
| 精排 | `BAAI/bge-reranker-large` CrossEncoder |
| 学习排序 | LightGBM LambdaMART (100 条标注数据训练) |
| 生成模型 | `Qwen/Qwen2.5-3B-Instruct` |
| 向量库 | ChromaDB |
| 后端 | FastAPI |
| 前端 | HTML / CSS / JavaScript |
| 文件解析 | pypdf / python-docx |

## 支持的合规政策 (30 条)

| 类别 | 政策 ID |
|------|---------|
| 个人信息保护 | PIPL-001 ~ PIPL-008 |
| 数据安全 | DSL-001 ~ DSL-006 |
| 网络安全 | CSL-001 ~ CSL-004 |
| AI 合规 | AI-001 ~ AI-004 |
| 员工行为规范 | EMP-001 ~ EMP-002 |
| 合同审核 | CONTRACT-001 ~ CONTRACT-002 |
| 信息披露 | DISC-001 |
| 安全管理 | SEC-001 ~ SEC-002 |
| 财务合规 | FIN-001 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 构建检索索引

```bash
python build_policy_rag.py
```

首次运行会下载 `BAAI/bge-large-zh-v1.5`（~1.3 GB）并生成：
- `chroma_db/` — 向量索引
- `bm25_index.pkl` — BM25 稀疏索引

### 3. 训练 LambdaMART 精排模型

```bash
python train_lambdamart.py
```

基于 `raw_policy/labeled_data.json` 中 100 条标注样例（80 训练 / 20 验证）训练，生成 `lambdamart_model.txt`，服务启动时自动加载。

### 4. 启动服务

```bash
python app.py
# 或
uvicorn app:app --reload --port 8000
```

访问：[http://127.0.0.1:8000](http://127.0.0.1:8000)

## 评估

运行检索质量评估（对比 5 种策略）：

```bash
python eval.py          # 全部 100 条，K=5
python eval.py --k 3    # 使用 K=3
python eval.py --split val   # 仅验证集（后 20 条）
```

### 当前评估结果 (K=5, 100 条样例)

| 策略 | R@5 | NDCG@5 | F1@5 |
|------|-----|--------|------|
| dense_only | 0.868 | 0.831 | 0.387 |
| bm25_only | 0.811 | 0.744 | 0.373 |
| hybrid_rrf | 0.863 | 0.796 | 0.389 |
| hybrid_reranker | 0.977 | 0.918 | 0.450 |
| **full_pipeline** | **0.963** | **0.948** | **0.442** |

- **R@5**：Top-5 结果中命中相关政策的比例（越高越好，避免漏报）
- **NDCG@5**：相关政策是否排在靠前位置（越高越好，影响 LLM 上下文质量）
- **F1@5**：精确率与召回率的调和平均

## 文件说明

```text
raw_policy/
  policy_knowledge.json     30 条中国企业合规政策
  labeled_data.json         100 条检索标注样例（含相关政策 ID ground truth）

build_policy_rag.py         构建 ChromaDB 向量库 + BM25 索引
train_lambdamart.py         训练 LambdaMART 精排模型
eval.py                     检索质量评估（P/R/NDCG/F1 @ K）
rag_models.py               完整 RAG 管线：检索 / 融合 / 精排 / 生成
file_parser.py              解析 txt / pdf / doc / docx
app.py                      FastAPI 后端
static/                     前端页面

chroma_db/                  运行 build_policy_rag.py 后生成（不纳入 git）
bm25_index.pkl              运行 build_policy_rag.py 后生成（不纳入 git）
lambdamart_model.txt        运行 train_lambdamart.py 后生成（不纳入 git）
uploads/                    上传文件临时目录（不纳入 git）
```

## 注意事项

- 首次运行会下载三个模型：`bge-large-zh-v1.5`（~1.3 GB）、`bge-reranker-large`（~2.2 GB）、`Qwen2.5-3B-Instruct`（~6 GB），请确保网络畅通。
- 推荐 GPU 环境运行；CPU 可用但生成速度较慢。
- `.doc` 旧版 Word 格式为 best-effort 解析，建议转为 `.docx` 或 PDF 后上传。
- 修改政策库后需重新运行 `build_policy_rag.py` 和 `train_lambdamart.py`。
