# Policy RAG Audit System - Qwen Version

这是一个 Python 版中国企业政策审核 RAG demo，workflow 复用 RDR2 项目的结构：

```text
policy JSON -> chunk -> embedding -> ChromaDB -> 上传文件/输入文本 -> retrieve policy -> rerank -> Qwen 生成审核报告
```

## 技术栈

- Python
- FastAPI
- ChromaDB
- SentenceTransformers: `sentence-transformers/all-MiniLM-L6-v2`
- CrossEncoder reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Qwen: `Qwen/Qwen2.5-3B-Instruct`
- pypdf / python-docx
- HTML/CSS/JavaScript 前端

## 运行

```bash
cd policy_rag_qwen
pip install -r requirements.txt
python build_policy_rag.py
uvicorn app:app --reload --port 8000
```

打开：

```text
http://127.0.0.1:8000
```

## 文件说明

```text
raw_policy/policy_knowledge.json   中国企业合规政策 JSON 数据
build_policy_rag.py                构建 ChromaDB 向量库
rag_models.py                      Qwen + embedding + retrieve + rerank + audit
file_parser.py                     解析 txt/pdf/doc/docx
app.py                             FastAPI 后端
static/                            前端页面
chroma_db/                         运行 build 后生成
uploads/                           上传文件临时目录
```

## 注意

首次运行会下载 Qwen、embedding model 和 reranker。Qwen2.5-3B 建议使用有 GPU 的环境；CPU 也能运行但会较慢。

`.doc` 是旧版 Word 二进制格式，本 demo 做 best-effort 文本读取。正式项目建议用 LibreOffice 先转换成 `.docx` 或 PDF。
