"""CPU semantic retrieval using an already installed model; no API or download."""
from functools import lru_cache
import json
from threading import RLock
from langchain_core.embeddings import Embeddings

MODEL = "BAAI/bge-small-zh-v1.5"
PREFIX = "为这个句子生成表示以用于检索相关文章："


class LocalEmbeddings(Embeddings):
    def __init__(self, cache_dir, model=MODEL):
        from fastembed import TextEmbedding
        self.model = TextEmbedding(model_name=model, cache_dir=str(cache_dir), threads=2,
            providers=["CPUExecutionProvider"], local_files_only=True)
        self.lock = RLock()
        self.signature = json.dumps({"model":model,"runtime":"fastembed-onnx-cpu","query_prefix":PREFIX},sort_keys=True)

    def embed_documents(self, texts):
        with self.lock:
            return [v.tolist() for v in self.model.embed(texts,batch_size=16)]

    @lru_cache(maxsize=512)
    def _query(self, text):
        with self.lock:
            return tuple(next(self.model.embed([PREFIX+text])).tolist())

    def embed_query(self, text):
        return list(self._query(text))
