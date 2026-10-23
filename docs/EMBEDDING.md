# Embedding配置与检索对照

Embedding模型把资料和问题编码成向量，供语义相似度检索使用。它不负责生成聊天回答；当前DeepSeek聊天接口不能直接作为Embedding接口。

## 配置方式

项目支持OpenAI兼容的Embedding服务。以阿里云百炼为例：在控制台选择服务地域/工作空间，开通可用的文本向量模型，创建该地域的API Key，然后从该模型的兼容接口文档复制Base URL。Key和地域必须对应，不要把Key发送到聊天或提交到Git。

在项目本机`.env`填写：

```dotenv
EMBEDDING_API_KEY=在本机填写
EMBEDDING_BASE_URL=从所选地域工作空间的兼容接口文档复制
EMBEDDING_MODEL=text-embedding-v4
```

模型名应以该地域实际可用列表为准，`text-embedding-v4`只是可选示例。官方文档目前区分工作空间域名；不把其他地域的URL或聊天模型路径直接套过来。服务可能计费，开通后先用少量资料验证。

参考：[百炼Embedding兼容接口](https://www.alibabacloud.com/help/en/model-studio/embedding-interfaces-compatible-with-openai)、[获取API Key](https://www.alibabacloud.com/help/en/model-studio/get-api-key)。

## 同一索引比较三种检索

```powershell
.\.venv\Scripts\python.exe -m app.knowledge_cli build --input your-reviewed-phones.jsonl --method vector --output runs/phones-vector-new.json
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/phones-vector-new.json --phone "完整机型" --need "你的问题" --method bm25
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/phones-vector-new.json --phone "完整机型" --need "你的问题" --method vector --allow-embedding
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/phones-vector-new.json --phone "完整机型" --need "你的问题" --method hybrid --allow-embedding
```

构建时生成文档向量并保存；查询时只生成问题向量，加载索引不会重复生成全部文档向量。RRF按BM25和向量排名融合，不直接相加两个尺度不同的分数，返回片段编号与分量排名。

聊天可配置`CHAT_KNOWLEDGE_INDEX_PATH`和`CHAT_RETRIEVAL_METHOD=hybrid`。默认仍使用随包BM25；引入外部索引后应重新启动服务。向量接口和模型改变时重建索引。

## 已运行的本地真实Embedding对照

2026-10-21已完成20款手机、240个官方规格片段、60题的BM25/真实向量/RRF对照和型号过滤消融。使用本地BGE中文预训练模型、FastEmbed和CPU，无需付费Embedding服务。方法、结果和失败案例见[新对照记录](../evaluation/RAG_BENCHMARK_RESULTS.md)。标签仍为AI初标，独立人工盲审待用户完成。

本地模型仅用于独立实验，未改变首页默认检索；正式服务的向量路径仍按上方配置服务和索引。可选实验依赖用`pip install ".[eval]"`安装，默认评测不会自动下载模型。

新实验应冻结语料和切分，标注相关片段，记录过滤后的候选数量；同一问题分别测试BM25、向量与RRF，计算Recall@k、MRR、无资料表现及耗时。完整型号过滤单独做消融。跨机型筛选另测预算、版本和日期约束，不能只用相似度筛选价格。独立人工标注部分样本，再核对分歧；不把一致率当作事实准确率。
