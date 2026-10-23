"""Build and inspect a local knowledge index; default retrieval is network-free."""

import argparse
import json
from pathlib import Path
import sys

from .knowledge import KnowledgeBase, build_index, save_index, load_index, vectorize_index
from .config import get_settings


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="手机知识库导入与检索")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="从 JSONL 文档导入、切分并建立索引")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--method", choices=("bm25", "vector"), default="bm25",
                       help="bm25 完全本地；vector 会调用配置的 Embedding 服务")
    query = commands.add_parser("query", help="检查实际召回片段，不调用分析模型")
    query.add_argument("--index", type=Path, required=True)
    query.add_argument("--phone", required=True)
    query.add_argument("--need", required=True)
    query.add_argument("--method", choices=("bm25", "vector", "hybrid"),
                       help="复用同一向量索引比较 BM25、向量或 RRF 混合检索")
    query.add_argument("--allow-embedding", action="store_true", help="允许向量查询访问配置的 Embedding 服务")
    vectorize = commands.add_parser('vectorize',help='为已有索引添加向量，保留来源编号；默认只允许本地模型')
    vectorize.add_argument('--input',type=Path,required=True)
    vectorize.add_argument('--output',type=Path,required=True)
    vectorize.add_argument('--allow-remote',action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command in {"build",'vectorize'}:
            if args.output.exists():
                raise FileExistsError("索引已存在，请选择新文件名")
            if args.command=='vectorize':
                if get_settings().embedding_provider.lower()!='local' and not args.allow_remote:
                    raise ValueError('远程Embedding需要显式--allow-remote')
                index=vectorize_index(load_index(args.input))
            else:
                index = build_index(args.input, args.method)
            save_index(index, args.output)
            result = {"index": str(args.output), "method": index.method, "chunks": len(index.chunks)}
        else:
            base = KnowledgeBase.from_path(args.index, allow_embeddings=args.allow_embedding, search_method=args.method)
            result = base.as_tool().invoke({"phone_name": args.phone, "need": args.need})
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as error:
        print(f"知识库操作失败：{type(error).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
