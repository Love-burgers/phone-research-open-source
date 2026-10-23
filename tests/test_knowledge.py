"""Conversation-side ingestion, retrieval, filtering and persisted-vector tests."""
import json
from pathlib import Path
from langchain_core.embeddings import Embeddings
import pytest
from app.knowledge import KnowledgeBase, KnowledgeIndex, build_index, load_index, save_index
from app.knowledge_cli import main as knowledge_main
SAMPLE = Path(__file__).parents[1] / "app" / "knowledge_samples" / "phones.jsonl"

def write_documents(tmp_path, records):
    path = tmp_path / "documents.jsonl"
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")
    return path

def document(phone="Phone A", text="续航测试连续播放视频 10 小时。", url="https://example.com/review"):
    return dict(phone_name=phone, title="测试来源", source_url=url, text=text)

class TinyEmbeddings(Embeddings):
    """Deliberately fake vectors for deterministic vector-path tests only."""
    def __init__(self):
        self.document_calls = self.query_calls = 0
    def embed_documents(self, texts):
        self.document_calls += 1
        return [[1., 0.] if "续航" in text else [0., 1.] for text in texts]
    def embed_query(self, text):
        self.query_calls += 1
        return [1., 0.] if text == "电量维持时间" else [0., 1.]

def test_corrupt_vector_dimensions_and_nonfinite_values_are_rejected(tmp_path):
    embeddings = TinyEmbeddings()
    payload = build_index(SAMPLE, "vector", embeddings=embeddings, embedding_signature="fake").model_dump()
    payload["chunks"][0]["vector"] = [1., 2., 3.]
    with pytest.raises(ValueError):
        KnowledgeIndex.model_validate(payload)
    payload["chunks"][0]["vector"] = [float("nan"), 0.]
    with pytest.raises(ValueError):
        KnowledgeIndex.model_validate(payload)

def test_duplicate_documents_do_not_create_duplicate_chunks(tmp_path):
    index = build_index(write_documents(tmp_path, [document(), document()]))
    assert len(index.chunks) == 1

def test_import_chunking_persistence_and_exact_phone_filter(tmp_path):
    source = write_documents(tmp_path, [document(text="续航测试内容。" * 180), document("Phone A Pro"), document("Phone B")])
    index = build_index(source)
    assert len(index.chunks) > 3
    assert all(len(chunk.text) <= 600 for chunk in index.chunks)
    path = tmp_path / "index.json"
    save_index(index, path)
    base = KnowledgeBase.from_path(path)
    hits = base.as_tool().invoke({"phone_name": "  phone   a  ", "need": "续航"})
    assert hits and all(item["phone_name"] == "Phone A" for item in hits)
    assert all(item["chunk_id"] and item["document_id"] for item in hits)
    assert load_index(path) == index
    with pytest.raises(FileExistsError):
        save_index(index, path)

@pytest.mark.parametrize("records", [[], [document(text=" ")], [document(url="javascript:alert(1)")], [document(url="https://user:secret@example.com/review")]])
def test_invalid_import_is_rejected(tmp_path, records):
    with pytest.raises(ValueError):
        build_index(write_documents(tmp_path, records))

@pytest.mark.parametrize("phone,need", [("Unknown", "续航"), ("Phone A", "防水等级"), ("Phone A", " ")])
def test_missing_or_irrelevant_knowledge_returns_empty(phone, need):
    base = KnowledgeBase(build_index(SAMPLE))
    assert base.as_tool().invoke({"phone_name": phone, "need": need}) == []

def test_vector_index_persists_and_queries_without_reembedding_documents(tmp_path):
    source = write_documents(tmp_path, [document(), document(text="拍照样张细节", url="https://example.com/camera")])
    embeddings = TinyEmbeddings()
    index = build_index(source, "vector", embeddings=embeddings, embedding_signature="fake-test-model")
    path = tmp_path / "vector.json"
    save_index(index, path)
    base = KnowledgeBase(load_index(path), embeddings)
    hits = base.as_tool().invoke({"phone_name": "Phone A", "need": "电量维持时间"})
    assert len(hits) == 1 and "续航" in hits[0]["text"]
    assert embeddings.document_calls == 1 and embeddings.query_calls == 1
    with pytest.raises(ValueError, match="离线"):
        KnowledgeBase.from_path(path, allow_embeddings=False)

def test_chat_knowledge_cli_build_and_query(tmp_path, capsys):
    index = tmp_path / "index.json"
    assert knowledge_main(["build", "--input", str(SAMPLE), "--output", str(index)]) == 0
    capsys.readouterr()
    assert knowledge_main(["query", "--index", str(index), "--phone", "Phone A", "--need", "续航"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["chunk_id"]
