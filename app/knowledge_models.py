"""Knowledge schemas; independent of ingestion, retrieval and providers."""

import math
from typing import Literal
from urllib.parse import urlsplit
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    phone_name: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=240)
    source_url: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1, max_length=20000)
    checked_on: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    published_on: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    market: str = "地区未确认"
    evidence_kind: Literal["spec", "review", "price", "unknown"] = "unknown"
    reference_price: int | None = Field(default=None, ge=500, le=100000)
    price_variant: str | None = None

    @field_validator("source_url")
    @classmethod
    def check_source(cls, value):
        # A citation is stored, never fetched during import. Do not resolve DNS.
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("来源必须为无认证信息的 HTTP/HTTPS URL")
        return value

class KnowledgeChunk(KnowledgeDocument):
    text: str = Field(min_length=1, max_length=800)
    document_id: str = Field(min_length=1, max_length=80)
    chunk_id: str = Field(min_length=1, max_length=80)
    vector: list[float] | None = Field(default=None, min_length=1, max_length=4096)

class KnowledgeIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    method: Literal["bm25", "vector"] = "bm25"
    embedding_signature: str | None = None
    chunks: list[KnowledgeChunk] = Field(min_length=1, max_length=1024)

    @model_validator(mode="after")
    def check_index(self):
        if len({chunk.chunk_id for chunk in self.chunks}) != len(self.chunks):
            raise ValueError("知识片段编号重复")
        vectors = [chunk.vector for chunk in self.chunks]
        if self.method == "vector":
            if not self.embedding_signature or any(vector is None for vector in vectors):
                raise ValueError("向量索引缺少模型信息或向量")
            if len({len(vector) for vector in vectors}) != 1:
                raise ValueError("向量维度不一致")
            if any(not all(math.isfinite(x) for x in vector) or not any(vector) for vector in vectors):
                raise ValueError("向量必须为有限非零数值")
        elif self.embedding_signature is not None or any(vector is not None for vector in vectors):
            raise ValueError("BM25 索引不能包含向量")
        return self
