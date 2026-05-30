"""运行生产化 ingestion demo。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ingestion_pipeline import IngestionPipeline, StructureAwareChunker


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行文档加载与结构化 chunking demo")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "sample_docs")
    parser.add_argument("--output", type=Path, default=Path("/tmp/rag-production-ingestion"))
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--overlap-tokens", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """构建管线并输出运行摘要。"""

    args = parse_args()
    chunker = StructureAwareChunker(
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    pipeline = IngestionPipeline(args.input, args.output, chunker)
    stats = pipeline.run()

    print("Ingestion 完成：")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"chunks:   {pipeline.chunks_path}")
    print(f"manifest: {pipeline.manifest_path}")


if __name__ == "__main__":
    main()

