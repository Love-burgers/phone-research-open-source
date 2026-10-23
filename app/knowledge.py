"""Small persistent phone knowledge base, LangChain retrieval and cited evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal

from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from rank_bm25 import BM25Plus

from .config import EMBEDDING_TIMEOUT_SECONDS, get_settings
from .retrieval import focus_aliases

from .knowledge_models import KnowledgeDocument, KnowledgeChunk, KnowledgeIndex
from .provider_embeddings import QuotaEmbeddings


def _name(value: str) -> str:
    return " ".join(value.split()).casefold()


def configured_embeddings() -> tuple[Embeddings, str]:
    """Embedding configuration is separate from the DeepSeek chat endpoint."""
    settings = get_settings()
    if settings.embedding_provider.lower() == "local":
        from .local_embeddings import LocalEmbeddings, MODEL
        client = LocalEmbeddings(settings.embedding_local_cache_dir, settings.embedding_model or MODEL)
        return client, client.signature
    key = settings.embedding_api_key.get_secret_value().strip() if settings.embedding_api_key else ""
    base_url, model = settings.embedding_base_url.strip(), settings.embedding_model.strip()
    if not all((key, base_url, model)):
        raise ValueError("向量检索需要 EMBEDDING_API_KEY、EMBEDDING_BASE_URL 和 EMBEDDING_MODEL")
    signature = json.dumps({"base_url": base_url.rstrip("/"), "model": model}, sort_keys=True)
    client = OpenAIEmbeddings(api_key=key, base_url=base_url, model=model,
                              check_embedding_ctx_length=False,chunk_size=64,max_retries=0, request_timeout=EMBEDDING_TIMEOUT_SECONDS)
    return QuotaEmbeddings(client), signature


def build_index(input_path: Path, method: Literal["bm25", "vector"] = "bm25", *,
                embeddings: Embeddings | None = None, embedding_signature: str | None = None) -> KnowledgeIndex:
    if method not in {"bm25", "vector"}:
        raise ValueError("未知检索方式")
    if input_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("输入资料超过 2 MiB")
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80,
                                               separators=["\n\n", "\n", "。", "；", " ", ""])
    chunks, seen = [], set()
    for line in input_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = KnowledgeDocument.model_validate_json(line)
        document_id = hashlib.sha256(f"{_name(item.phone_name)}\n{item.source_url}".encode()).hexdigest()[:24]
        documents = splitter.split_documents([Document(page_content=item.text)])
        for document in documents:
            text = document.page_content.strip()
            chunk_id = hashlib.sha256(f"{document_id}\n{text}".encode()).hexdigest()[:24]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunks.append(KnowledgeChunk(**{**item.model_dump(), "text": text},
                                         document_id=document_id, chunk_id=chunk_id))
    # Validate bounds before making any embedding request.
    draft = KnowledgeIndex(chunks=chunks)
    if method == "bm25":
        return draft
    return vectorize_index(draft,embeddings=embeddings,embedding_signature=embedding_signature)


def vectorize_index(index, *, embeddings=None, embedding_signature=None):
    """Add vectors to existing validated records, preserving provenance/IDs."""
    if embeddings is None:
        embeddings, embedding_signature = configured_embeddings()
    if not embedding_signature:
        raise ValueError("向量索引需要明确的模型签名")
    vectors = embeddings.embed_documents([chunk.text for chunk in index.chunks])
    if len(vectors) != len(index.chunks):
        raise ValueError("Embedding 返回的数量不匹配")
    return KnowledgeIndex(method="vector", embedding_signature=embedding_signature,
                          chunks=[KnowledgeChunk(**{**chunk.model_dump(), "vector": vector})
                                  for chunk, vector in zip(index.chunks, vectors)])


def save_index(index: KnowledgeIndex, path: Path) -> None:
    text = index.model_dump_json(indent=2) + "\n"
    if len(text.encode("utf-8")) > 20 * 1024 * 1024:
        raise ValueError("知识索引超过 20 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(text)


def load_index(path: Path) -> KnowledgeIndex:
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("知识索引超过 20 MiB")
    return KnowledgeIndex.model_validate_json(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> list[str]:
    """Chinese bigrams and Latin words; no English-only whitespace tokenization."""
    tokens = []
    for part in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.casefold()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part) and len(part) > 1:
            tokens.extend(part[i:i + 2] for i in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def _document(chunk: KnowledgeChunk, score: float) -> Document:
    return Document(id=chunk.chunk_id, page_content=chunk.text, metadata={
        "phone_name": chunk.phone_name, "title": chunk.title, "source_url": chunk.source_url,
        "document_id": chunk.document_id, "chunk_id": chunk.chunk_id, "retrieval_score": score,
        "checked_on": chunk.checked_on, "published_on": chunk.published_on,
        "market": chunk.market, "evidence_kind": chunk.evidence_kind,
        "reference_price": chunk.reference_price, "price_variant": chunk.price_variant,
    })


class _StoredEmbeddings(Embeddings):
    """Rehydrate stored vectors without re-embedding documents on each load."""

    def __init__(self, chunks, delegate):
        self.chunks, self.delegate = chunks, delegate

    def embed_documents(self, texts):
        if texts != [chunk.text for chunk in self.chunks]:
            raise ValueError("索引文本与向量不匹配")
        return [chunk.vector for chunk in self.chunks]

    def embed_query(self, text):
        vector = self.delegate.embed_query(text)
        if len(vector) != len(self.chunks[0].vector) or not all(math.isfinite(x) for x in vector) or not any(vector):
            raise ValueError("查询向量与索引不兼容")
        return vector


class PhoneKnowledgeRetriever(BaseRetriever):
    index: KnowledgeIndex
    phone_name: str
    k: int = Field(default=4, ge=1, le=1024)
    embeddings: Any = Field(default=None, exclude=True)
    min_vector_score: float = Field(default=0.35, ge=-1, le=1)
    search_method: Literal["bm25", "vector", "hybrid"] | None = None
    filter_phone: bool = True
    candidate_k: int = Field(default=20, ge=8, le=1024)

    def _get_relevant_documents(self, query: str, *, run_manager):
        chunks = [chunk for chunk in self.index.chunks
                  if not self.filter_phone or _name(chunk.phone_name) == _name(self.phone_name)]
        if not chunks or not query.strip():
            return []
        method = self.search_method or self.index.method
        vector_documents = []
        if method in {"vector", "hybrid"}:
            if self.index.method != "vector":
                raise ValueError("向量/混合检索需要带持久化向量的索引")
            if self.embeddings is None:
                raise ValueError("向量检索未配置 Embedding")
            store = InMemoryVectorStore(_StoredEmbeddings(chunks, self.embeddings))
            store.add_documents([_document(chunk, 0) for chunk in chunks])
            vector_documents = [Document(id=doc.id, page_content=doc.page_content,
                             metadata={**doc.metadata, "retrieval_score": float(score)})
                    for doc, score in store.similarity_search_with_score(
                        query, k=self.candidate_k if method == "hybrid" else self.k)
                    if score >= self.min_vector_score]
            if method == "vector":
                return vector_documents
        expanded = " ".join([query, *(word for group in focus_aliases(query) for word in group)])
        tokens = tokenize(expanded)
        corpus = [tokenize(chunk.text) for chunk in chunks]
        keyword_documents = []
        if tokens and any(corpus):
            scores = BM25Plus(corpus, delta=0).get_scores(tokens)
            ranked = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i].chunk_id))
            keyword_documents = [_document(chunks[i], float(scores[i])) for i in ranked[
                :self.candidate_k if method == "hybrid" else self.k] if scores[i] > 0]
        return reciprocal_rank_fusion(keyword_documents, vector_documents, self.k) if method == "hybrid" else keyword_documents


def reciprocal_rank_fusion(keyword_documents, vector_documents, k, rank_constant=60):
    """Fuse ranks by stable chunk ID; BM25/cosine scores have different scales."""
    scores, documents, ranks = {}, {}, {}
    for label, ranked in (("bm25", keyword_documents), ("vector", vector_documents)):
        seen = set()
        for rank, document in enumerate(ranked, 1):
            chunk_id = document.metadata["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (rank_constant + rank)
            documents[chunk_id] = document
            ranks.setdefault(chunk_id, {})[label] = rank
    ordered = sorted(scores, key=lambda cid: (-scores[cid], cid))[:k]
    return [Document(id=cid, page_content=documents[cid].page_content,
                     metadata={**documents[cid].metadata, "retrieval_score": scores[cid],
                               "component_ranks": ranks[cid]}) for cid in ordered]


class KnowledgeBase:
    def coverage(self):
        kinds = ('spec','price','review','unknown')
        dates = sorted(c.checked_on for c in self.index.chunks if c.checked_on)
        return {'models':len({c.phone_name for c in self.index.chunks}), 'chunks':len(self.index.chunks),
            'chunks_by_kind':{kind:sum(c.evidence_kind==kind for c in self.index.chunks) for kind in kinds},
            'models_by_kind':{kind:len({c.phone_name for c in self.index.chunks if c.evidence_kind==kind}) for kind in kinds},
            'oldest_checked_on':dates[0] if dates else None,'newest_checked_on':dates[-1] if dates else None,
            'scope':'Dated source records; coverage is not factual accuracy or current price availability.'}

    def __init__(self, index: KnowledgeIndex, embeddings: Embeddings | None = None, *,
                 search_method: Literal["bm25", "vector", "hybrid"] | None = None, min_vector_score=.35):
        self.index, self.embeddings = index, embeddings
        self.search_method = search_method
        self.min_vector_score = min_vector_score
        self.path = None
        from threading import RLock
        self.write_lock = RLock()

    @classmethod
    def from_path(cls, path: Path, *, allow_embeddings: bool = True, search_method=None):
        index = load_index(path)
        embeddings = None
        if index.method == "vector" and search_method != "bm25":
            if not allow_embeddings and get_settings().embedding_provider.lower() != "local":
                raise ValueError("离线模式只允许 BM25 知识索引；向量查询需要显式联网模式")
            embeddings, signature = configured_embeddings()
            if signature != index.embedding_signature:
                raise ValueError("Embedding 模型或接口与构建索引时不一致，请重新构建")
        base = cls(index, embeddings, search_method=search_method,
                   min_vector_score=get_settings().chat_vector_min_score)
        base.path = path
        return base

    def discover(self, query):
        """Search across models; the agent still checks budget and exclusions."""
        # Diversification must happen before the document cutoff. As the corpus
        # grows, 64 highly ranked chunks from a few phones can otherwise hide an
        # entire eligible price tier. This scans only the bounded local index;
        # the tool still returns a small, condition-checked set to the model.
        size = len(self.index.chunks)
        retriever = PhoneKnowledgeRetriever(index=self.index, phone_name="", filter_phone=False,
            embeddings=self.embeddings, search_method=self.search_method, k=size, candidate_k=max(8,size),
            min_vector_score=self.min_vector_score)
        documents=retriever.invoke(query)
        first,rest,seen=[],[],set()
        for doc in documents:
            name=doc.metadata['phone_name']
            if name in seen:rest.append(doc)
            else:first.append(doc);seen.add(name)
        return [{"text":doc.page_content,**doc.metadata} for doc in [*first,*rest]]

    def remember_sources(self, sources, catalog):
        # Only the explicitly configured local model may embed automatically.
        # Remote embedding providers retain their explicit import workflow.
        from .local_embeddings import LocalEmbeddings
        if self.path is None or not isinstance(self.embeddings,LocalEmbeddings):
            return {"added_chunks":0}
        from .knowledge_ingest import source_documents
        import tempfile
        docs = source_documents(sources,catalog)
        if not docs:
            return {"added_chunks":0}
        with self.write_lock, tempfile.TemporaryDirectory(dir=self.path.parent) as folder:
            source_path = Path(folder)/"incoming.jsonl"
            source_path.write_text("\n".join(d.model_dump_json() for d in docs),encoding="utf-8")
            incoming = build_index(source_path).chunks
            existing = {c.chunk_id for c in self.index.chunks}
            new = [c for c in incoming if c.chunk_id not in existing]
            if not new:
                return {"added_chunks":0}
            # Keep one current offer per source/version; old source records remain
            # in the conversation and append-only import history.
            replaced = {(c.phone_name,c.source_url,c.price_variant) for c in new if c.evidence_kind=="price"}
            kept = [c for c in self.index.chunks if c.evidence_kind!="price" or
                    (c.phone_name,c.source_url,c.price_variant) not in replaced]
            if len(kept)+len(new)>1024:
                return {"added_chunks":0,"status":"capacity_limit"}
            vectors = self.embeddings.embed_documents([c.text for c in new])
            if len(vectors) != len(new):
                raise ValueError("新增资料与向量数量不匹配，保留原索引")
            updated = KnowledgeIndex(method="vector",embedding_signature=self.index.embedding_signature,
                chunks=[*kept,*(c.model_copy(update={"vector":v}) for c,v in zip(new,vectors))])
            saved = Path(folder)/"index.json"
            save_index(updated,saved)
            saved.replace(self.path)
            self.index = updated
            with self.path.with_suffix(".updates.jsonl").open("a",encoding="utf-8") as history:
                for doc in docs:
                    history.write(doc.model_dump_json()+"\n")
            return {"added_chunks":len(new),"embedding_api_requests":0}

    def as_tool(self):
        @tool("search_phone_knowledge")
        def search_phone_knowledge(phone_name: str, need: str) -> list[dict]:
            """Retrieve relevant source passages for one exact phone and a user need."""
            retriever = PhoneKnowledgeRetriever(index=self.index, phone_name=phone_name, embeddings=self.embeddings,
                                                 search_method=self.search_method,min_vector_score=self.min_vector_score)
            return [{"text": doc.page_content, **doc.metadata} for doc in retriever.invoke(need)]
        return search_phone_knowledge
