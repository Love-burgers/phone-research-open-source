# 手机资料知识库 RAG

项目中的知识库负责保存资料片段与来源，检索负责找相关片段，RAG 将片段交给模型生成带引用的报告。LangChain 提供模型、切分器、Retriever 和工具接口，LangGraph 决定知识库之后是否需要补搜。设计依据见 [LangChain 检索说明](https://docs.langchain.com/oss/python/deepagents/retrieval) 与 [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)。

## 导入资料

输入为 UTF-8 JSONL，每行一份文档，四个字段必填：phone_name、title、source_url、text。phone_name 必须和研究时的明确机型对应；每条文档只归属一款手机。跨机型文章需先分别摘录，不能把另一款的参数混入当前机型。

~~~json
{"phone_name":"具体手机型号","title":"来源标题与版本","source_url":"https://example.com/review","text":"这里替换为已核对来源的规格、评测或说明书正文。"}
~~~

当前导入的是文本 JSONL；PDF、网页或参数表应先提取并核对文本再导入，没有自动 PDF 解析或全站爬取。保留原链接，在正文注明地区版本、日期、测试条件和冲突；代码不会认证来源权威性或自动核验事实。内置 app/knowledge_samples/phones.jsonl 全是虚构 Phone A/B 教学资料。

~~~powershell
.\.venv\Scripts\python.exe -m app.knowledge_cli build --input my-phones.jsonl --output runs/my-knowledge.json
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/my-knowledge.json --phone "具体手机型号" --need "重视续航"
~~~

导入通过 LangChain Document 与 RecursiveCharacterTextSplitter 生成约600字符片段、80字符重叠，保留原来源和稳定的 document_id、chunk_id。重复来源的相同片段只保存一次。输入上限2 MiB、单条正文20000字符、索引最多1024个片段和20 MiB；这是适合本地小语料的边界，超出时拒绝，不自动截断。

默认 BM25 是本地关键词检索，中文使用二字词组、英文使用单词，并扩展已有关注点别名。先按明确机型过滤，再取最多4个有正相关分数的片段；没有相关词就返回空。Phone A 不会匹配 Phone A Pro。分数不是答案置信度，也没有跨型号别名归一化。

## 向量检索

本机当前配置启用本地BGE中文向量与BM25混合检索：39款、301片段（271参数、28版本价、2评测），来自保留的官方资料及已取得的官方正文。索引为runs/chat-knowledge-live-20261022/index-vector.json，原始历史索引保持原样。运行模型BAAI/bge-small-zh-v1.5的本地ONNX缓存，向量构建与查询不调用收费接口。需安装可选依赖`pip install -e ".[local]"`并事先准备模型缓存；服务以local_files_only加载，不自动下载。设置EMBEDDING_PROVIDER=local、EMBEDDING_LOCAL_CACHE_DIR、EMBEDDING_MODEL、CHAT_KNOWLEDGE_INDEX_PATH及CHAT_RETRIEVAL_METHOD=hybrid。没有这些配置时仍使用随包BM25。当前本地余弦门槛0.30是有限探针校准值，不能代表所有场景的准确率。

Agent可跨机型检索，也可按完整型号检索。联网取得的官方正文经型号、地区、证据类型和版本价格字段检查后增量写入本地索引，保留导入历史，重复片段不重建向量；搜索摘要和未审核第三方评测不自动吸收。当前索引上限1024片段。资料以参数为主，向量库无法代替缺失的实测；详见[实际运行与限制](../evaluation/SEMANTIC_COST_20261022.md)。

在 .env 或环境变量设置 EMBEDDING_API_KEY、EMBEDDING_BASE_URL、EMBEDDING_MODEL，使用兼容 Embeddings API 的独立服务。DeepSeek 聊天接口不自动提供 Embedding；不要直接把聊天模型名称填写成向量模型。

~~~powershell
.\.venv\Scripts\python.exe -m app.knowledge_cli build --input my-phones.jsonl --output runs/my-vectors.json --method vector
.\.venv\Scripts\python.exe -m app.knowledge_cli query --index runs/my-vectors.json --phone "具体手机型号" --need "电量能维持多久" --allow-embedding
~~~

使用远端Embedding配置时，构建将资料发送给服务，查询发送问题；本地提供方则只在本机计算。向量与片段一起保存为验证后的JSON，查询加载时不重新请求文档向量。模型与提供方签名必须和构建时一致；更换模型、地区版本或资料需构建新索引。

检索使用 LangChain InMemoryVectorStore 与余弦相似度，默认门槛0.35；这个门槛是启发式值，需要根据实际模型、语料和无相关资料样本调整。单元测试用假向量验证接线；另已运行真实本地BGE的20款/240片段/60题对照，独立人工盲审待完成，见[新检索对照](../evaluation/RAG_BENCHMARK_RESULTS.md)。

## 对话中的检索

默认使用随包BM25。设置CHAT_KNOWLEDGE_INDEX_PATH加载本工程的外部索引，CHAT_RETRIEVAL_METHOD选择bm25/vector/hybrid；向量服务配置变化后需重建。主页Agent选择何时检索或补搜，不按旧对比研究流程运行。

检索工具返回text、完整机型、chunk_id与source_url，回答节点按同机型检查引用。旧对比研究的命令和图调度器已移到独立comparison_app工程，不是本对话项目入口。

## 检查效果

~~~powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_knowledge --output runs/knowledge-evaluation.json
~~~

固定虚构语料检查召回率、精确率、机型隔离和无资料返回，独立验证不依赖分析模型。真实资料还需检查：问法变化是否漏召回、跨机型内容是否混入、片段是否保留测试条件、结论是否受到引用支持。引用编号检查只证明编号归属，不证明语义支持；关键词覆盖也不能证明事实正确。

私人文档与索引放在 runs/ 等忽略目录，不纳入安装包。报告包含实际召回片段，分享前检查内容。网页尚无登录与访问控制，本地知识库不是多用户权限隔离系统。
