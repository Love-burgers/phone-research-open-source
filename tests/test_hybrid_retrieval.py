from langchain_core.documents import Document
import pytest

from app.knowledge import PhoneKnowledgeRetriever, reciprocal_rank_fusion
from tests.test_knowledge import TinyEmbeddings, document, write_documents
from app.knowledge import build_index


def doc(cid):
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_uses_rank_not_incompatible_raw_scores_and_deduplicates():
    result = reciprocal_rank_fusion([doc("a"), doc("b")], [doc("b"), doc("c")], 3)
    assert [d.metadata["chunk_id"] for d in result] == ["b", "a", "c"]
    assert result[0].metadata["component_ranks"] == {"bm25": 2, "vector": 1}
    assert reciprocal_rank_fusion([], [], 4) == []


def test_hybrid_keeps_phone_filter_and_sources(tmp_path):
    embeddings = TinyEmbeddings()
    source = write_documents(tmp_path, [document(), document(text="拍照样张细节", url="https://example.com/camera"),
                                       {**document(), "phone_name": "Phone B"}])
    index = build_index(source, "vector", embeddings=embeddings, embedding_signature="fake")
    hits = PhoneKnowledgeRetriever(index=index, phone_name="Phone A", embeddings=embeddings,
                                    search_method="hybrid").invoke("电量维持时间")
    assert hits and all(doc.metadata["phone_name"] == "Phone A" for doc in hits)
    assert all(doc.metadata["source_url"] and doc.metadata["chunk_id"] for doc in hits)
    assert embeddings.document_calls == 1


def test_bm25_on_same_persisted_vector_index_requires_no_embeddings(tmp_path):
    source = write_documents(tmp_path, [document()])
    index = build_index(source, "vector", embeddings=TinyEmbeddings(), embedding_signature="fake")
    hits = PhoneKnowledgeRetriever(index=index, phone_name="Phone A", search_method="bm25").invoke("续航")
    assert hits


def test_hybrid_rejects_index_without_vectors(tmp_path):
    index = build_index(write_documents(tmp_path, [document()]))
    with pytest.raises(ValueError, match="持久化向量"):
        PhoneKnowledgeRetriever(index=index, phone_name="Phone A", search_method="hybrid").invoke("续航")
