"""Explicitly snapshot public official specs; never replace the application's corpus."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from bs4 import BeautifulSoup, NavigableString
import httpx

from app.knowledge import build_index, save_index


APPLE_TOPICS = {
    "dimensions": r"尺寸.*重量", "screen": r"显示屏", "chip": r"芯片",
    "camera": r"^摄像头$", "video": r"视频拍摄", "front": r"原深感摄像头",
    "battery": r"电源和电池", "wireless": r"MagSafe", "network": r"蜂窝网络",
    "water": r"防溅", "sim": r"SIM 卡", "box": r"包装内容",
}
ONEPLUS_TOPICS = {
    "dimensions": r"外观尺寸", "chip": r"处理平台|性能", "storage": r"存储|储存",
    "screen": r"^屏幕$", "battery": r"电池与充电|电池", "camera": r"相机|摄像头",
    "network": r"网络与连接", "audio": r"音频", "material": r"材质",
    "ports": r"接口", "os": r"操作系统", "box": r"包装清单",
}


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def sections(html, parser):
    soup = BeautifulSoup(html, "html.parser")
    if parser == "apple":
        result = []
        for header in soup.select("h3"):
            pieces = []
            for node in header.next_elements:
                if getattr(node, "name", None) in {"h2", "h3"}:
                    break
                if isinstance(node, NavigableString) and node.parent.name not in {"script", "style", "sup"}:
                    pieces.append(str(node))
            result.append((clean(header.get_text(" ", strip=True)), clean(" ".join(pieces))))
        return result
    result = []
    for block in soup.select(".spec-content"):
        title = block.select_one(".title")
        if title:
            result.append((clean(title.get_text(" ", strip=True)), clean(block.get_text(" ", strip=True))))
    return result


def prepare(sources, output, raw_directory=None):
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    stamp = datetime.now(timezone.utc).isoformat()

    def download(item):
        slug = re.sub(r"[^a-z0-9]+", "-", item["phone"].lower()).strip("-")
        if item["phone"].startswith("一加"):
            slug = "oneplus-" + slug
        cached = raw_directory / f"{slug}.html" if raw_directory else None
        if cached and cached.is_file():
            html, resolved_url = cached.read_bytes(), item["url"]
            retrieved_at = datetime.fromtimestamp(cached.stat().st_mtime, timezone.utc).isoformat()
            from_cache = True
        else:
            response = httpx.get(item["url"], timeout=35, follow_redirects=True,
                                 headers={"User-Agent": "PhoneResearchBenchmark/1.0 (bounded official-spec snapshot)"})
            response.raise_for_status()
            html, resolved_url = response.content, str(response.url)
            retrieved_at, from_cache = datetime.now(timezone.utc).isoformat(), False
        (raw / f"{slug}.html").write_bytes(html)
        soup = BeautifulSoup(html, "html.parser")
        title = clean(soup.h1.get_text(" ", strip=True)) if soup.h1 else ""
        # Fail instead of silently attaching another model's redirected specs.
        expected = clean(item["phone"]).replace(" ", "").lower()
        if expected not in title.replace(" ", "").lower():
            raise ValueError(f"wrong page title for {item['phone']}: {title}")
        found, docs = sections(html, item["parser"]), []
        patterns = APPLE_TOPICS if item["parser"] == "apple" else ONEPLUS_TOPICS
        for topic, pattern in patterns.items():
            matches = [(heading, text) for heading, text in found if re.search(pattern, heading)]
            if len(matches) != 1:
                raise ValueError(f"{item['phone']} {topic}: expected one section, found {len(matches)}")
            heading, text = matches[0]
            # Bounded original section excerpts; no repetition to inflate the corpus.
            excerpt = f"{item['phone']}｜{text}"[:580]
            docs.append({"phone_name": item["phone"], "title": heading, "source_url": resolved_url,
                         "text": excerpt, "checked_on": stamp[:10], "market": "中国大陆官方规格页",
                         "evidence_kind": "spec", "topic": topic})
        return docs, {**item, "resolved_url": resolved_url, "page_title": title, "reused_snapshot": from_cache,
                      "retrieved_at": retrieved_at, "raw_sha256": hashlib.sha256(html).hexdigest(), "raw_file": f"raw/{slug}.html"}

    with ThreadPoolExecutor(max_workers=3) as pool:
        fetched = list(pool.map(download, sources))
    documents = [doc for docs, _ in fetched for doc in docs]
    corpus = output / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({k: v for k, v in d.items() if k != "topic"}, ensure_ascii=False) + "\n"
                              for d in documents), encoding="utf-8")
    index = build_index(corpus)
    if len(index.chunks) != len(documents):
        raise ValueError("Each frozen section must produce exactly one chunk")
    save_index(index, output / "index-bm25.json")
    labels = [{"phone": doc["phone_name"], "topic": doc["topic"], "chunk_id": chunk.chunk_id}
              for doc, chunk in zip(documents, index.chunks)]
    manifest = {"schema_version": 1, "retrieved_at": stamp, "sources": [item for _, item in fetched],
                "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                "chunk_count": len(index.chunks), "phone_count": len(sources), "sections": labels,
                "label_provenance": "Codex section-level initial annotation; not independently human-reviewed",
                "scope": "Official specs only; no independent experience, price, availability or current market claims"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"phones": len(sources), "chunks": len(index.chunks), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=Path("evaluation/rag_benchmark_sources.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Explicitly allow bounded public-page downloads")
    parser.add_argument("--raw-directory", type=Path, help="Reuse previously downloaded snapshots when present")
    args = parser.parse_args()
    if not args.download:
        parser.error("Add --download to fetch the public sources; no downloads by default")
    print(json.dumps(prepare(json.loads(args.sources.read_text(encoding="utf-8"))["sources"], args.output, args.raw_directory), ensure_ascii=False))


if __name__ == "__main__":
    main()
