# Embedding 配置

Embedding 将资料和查询编码为向量，聊天模型负责生成回答，两者配置独立。首页默认使用随包无向量 BM25；设置聊天 Key 不会自动启用向量或下载模型。

## 本地 BGE（可选运行路径）

本地 BGE 已接入正式知识库构建、查询与聊天运行路径，不限于独立实验。安装可选依赖，并事先准备 `BAAI/bge-small-zh-v1.5` 的 FastEmbed/ONNX 缓存；运行以 `local_files_only` 加载，缺缓存报错，不自动下载。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[local]"
```

本机 `.env` 示例：

```dotenv
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_LOCAL_CACHE_DIR=runs/model-cache
CHAT_KNOWLEDGE_INDEX_PATH=runs/phones-vector-new.json
CHAT_RETRIEVAL_METHOD=hybrid
```

先为当前随包索引生成新的外部向量文件，再启动服务：

```powershell
.\.venv\Scripts\python.exe -m app.knowledge_cli vectorize --input app/knowledge_samples/selection_index.json --output runs/phones-vector-new.json
```

此命令默认只允许本地模型，保留片段ID、日期和来源。本地计算不请求收费接口，向量化不会补足资料或核验事实。39款/301片段与98款/455片段都是历史外部索引规模，不是当前默认；复制旧索引不等于刷新内容。

## 远程兼容接口

独立设置 `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`。按所选服务官方文档填写匹配的地域、工作空间、兼容Base URL与实际可用模型，不套用聊天接口，不提交密钥。构建会发送资料，查询会发送查询文本，可能产生费用。

```powershell
# 明确允许远程向量化；先核对本机配置与消费额度
.\.venv\Scripts\python.exe -m app.knowledge_cli vectorize --input app/knowledge_samples/selection_index.json --output runs/phones-remote-new.json --allow-remote
```

网页个人聊天Key不覆盖远程Embedding Key。远程批次计入本工程持久化模型调用与token上限；本地计算不计付费请求。

## 同一索引比较检索

```powershell
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/phones-vector-new.json --phone "完整机型" --need "你的问题" --method bm25
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/phones-vector-new.json --phone "完整机型" --need "你的问题" --method vector --allow-embedding
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/phones-vector-new.json --phone "完整机型" --need "你的问题" --method hybrid --allow-embedding
```

文档向量构建一次并保存，查询只编码查询文本。模型/提供方签名必须匹配构建时记录，切换模型需重建。RRF融合排名，不直接相加不同尺度的分数；相似度阈值不是答案置信度。

历史检索实验与人工评审未公开，本文不链接私有结果文件或展示实验/成本数字。新对照应冻结语料和切分，独立标注，核对完整型号过滤、无资料返回和来源支持；不能把召回率写成推荐准确率。运行机制见 [RAG](RAG.md)，当前资料见 [CORPUS](CORPUS.md)。
