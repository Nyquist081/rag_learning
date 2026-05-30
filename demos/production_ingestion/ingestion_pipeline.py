"""接近生产形态的文档加载与切分管线。

这个 demo 只使用 Python 标准库，重点展示生产 ingestion 的结构：

1. 扫描数据源。
2. 按文件类型选择 loader。
3. 解析成带结构的 block。
4. 递归切分并保留 overlap。
5. 生成可追溯 metadata。
6. 使用内容哈希跳过未变化文档。
7. 输出 JSONL chunk 索引和 manifest。

真实项目可以把 loader 替换为 Unstructured、Docling、LlamaParse 等解析器，
再把 JSONL sink 替换为 Elasticsearch、OpenSearch 或向量数据库。
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Protocol


PARSER_VERSION = "stdlib-loaders-v1"
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".html", ".htm", ".json"}
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class SourceDocument:
    """loader 输出的统一文档对象。"""

    doc_id: str
    source_uri: str
    content_hash: str
    updated_at: str
    blocks: list["Block"]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Block:
    """保留结构信息的文本块。

    loader 不直接产出最终 chunk。它先产出 block，避免标题、代码块、表格
    和 JSON 记录在切分前被拍平成一段难以追溯的纯文本。
    """

    kind: str
    text: str
    title_path: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """准备写入检索索引的最终 chunk。"""

    chunk_id: str
    doc_id: str
    source_uri: str
    title_path: tuple[str, ...]
    text: str
    chunk_index: int
    token_count: int
    content_hash: str
    updated_at: str
    parser_version: str
    chunker_version: str
    metadata: dict[str, object] = field(default_factory=dict)


class Loader(Protocol):
    """文件 loader 协议：每种 loader 都要输出统一 SourceDocument。"""

    def load(self, path: Path) -> SourceDocument:
        """读取单个文件并转换成结构化文档。"""


def sha256_text(text: str) -> str:
    """计算稳定内容哈希，用于增量更新和去重。"""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_iso_from_mtime(path: Path) -> str:
    """把文件修改时间转成 UTC ISO-8601 字符串。"""

    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def stable_doc_id(path: Path, root: Path) -> str:
    """用相对路径生成稳定 doc_id。

    生产系统也可以使用数据库主键、对象存储 key 或业务侧文档 id。
    """

    relative = path.relative_to(root).as_posix()
    return sha256_text(relative)[:16]


class MarkdownLoader:
    """解析 Markdown，保留标题层级、代码块和表格。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, path: Path) -> SourceDocument:
        """读取 Markdown 文件并转换为结构化 blocks。"""

        text = path.read_text(encoding="utf-8")
        blocks = self._parse_blocks(text)
        return build_source_document(path, self.root, text, blocks, {"format": "markdown"})

    def _parse_blocks(self, text: str) -> list[Block]:
        """按标题、代码块、表格和段落解析 Markdown。"""

        title_path: list[str] = []
        blocks: list[Block] = []
        paragraph: list[str] = []
        fenced_code: list[str] = []
        table_rows: list[str] = []
        in_code = False

        def flush_paragraph() -> None:
            """把累计普通段落写入 blocks。"""

            if paragraph:
                blocks.append(Block("paragraph", "\n".join(paragraph), tuple(title_path)))
                paragraph.clear()

        def flush_table() -> None:
            """把连续 Markdown 表格作为整体写入 blocks。"""

            if table_rows:
                blocks.append(Block("table", "\n".join(table_rows), tuple(title_path)))
                table_rows.clear()

        for line in text.splitlines():
            if line.strip().startswith("```"):
                flush_paragraph()
                flush_table()
                fenced_code.append(line)
                if in_code:
                    blocks.append(Block("code", "\n".join(fenced_code), tuple(title_path)))
                    fenced_code.clear()
                in_code = not in_code
                continue

            if in_code:
                fenced_code.append(line)
                continue

            heading = HEADING_RE.match(line)
            if heading:
                flush_paragraph()
                flush_table()
                level = len(heading.group(1))
                title = heading.group(2).strip()
                title_path[:] = title_path[: level - 1]
                title_path.append(title)
                continue

            if "|" in line and line.strip().startswith("|"):
                flush_paragraph()
                table_rows.append(line)
                continue

            flush_table()
            if line.strip():
                paragraph.append(line.strip())
            else:
                flush_paragraph()

        flush_paragraph()
        flush_table()
        if fenced_code:
            blocks.append(Block("code", "\n".join(fenced_code), tuple(title_path)))
        return blocks


class TextLoader:
    """加载纯文本，并按空行保留段落结构。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, path: Path) -> SourceDocument:
        """读取纯文本文件。"""

        text = path.read_text(encoding="utf-8")
        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
        blocks = []
        title_path: tuple[str, ...] = ()
        for paragraph in paragraphs:
            # 纯文本没有显式标题标记。短单行段落通常是章节标题，
            # 将它挂到后续正文 metadata，而不是单独生成低价值 chunk。
            if "\n" not in paragraph and len(tokenize(paragraph)) <= 8 and not re.search(r"[.!?。！？]$", paragraph):
                title_path = (paragraph,)
                continue
            blocks.append(Block("paragraph", paragraph, title_path))
        return build_source_document(path, self.root, text, blocks, {"format": "text"})


class _HTMLBlockParser(HTMLParser):
    """提取 HTML 标题、段落、列表和代码文本。"""

    BLOCK_TAGS = {"p", "li", "pre", "code", "blockquote", "td", "th"}

    def __init__(self) -> None:
        super().__init__()
        self.title_path: list[str] = []
        self.blocks: list[Block] = []
        self._tag: str | None = None
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """遇到结构标签时开始累计文本。"""

        if tag in self.BLOCK_TAGS or re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._tag = tag

    def handle_endtag(self, tag: str) -> None:
        """结构标签结束时输出 block。"""

        if tag == self._tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        """累计 HTML 标签中的可见文本。"""

        if self._tag:
            self._buffer.append(data)

    def _flush(self) -> None:
        """把当前 HTML 文本缓存转换为 block。"""

        if not self._tag:
            return
        text = html.unescape(" ".join(self._buffer)).strip()
        if text:
            if re.fullmatch(r"h[1-6]", self._tag):
                level = int(self._tag[1])
                self.title_path[:] = self.title_path[: level - 1]
                self.title_path.append(text)
            else:
                self.blocks.append(Block(self._tag, text, tuple(self.title_path)))
        self._tag = None
        self._buffer.clear()


class HTMLLoader:
    """加载 HTML，丢弃样式脚本并保留主要结构元素。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, path: Path) -> SourceDocument:
        """读取 HTML 文件并转换为结构化 blocks。"""

        text = path.read_text(encoding="utf-8")
        parser = _HTMLBlockParser()
        parser.feed(text)
        parser.close()
        return build_source_document(path, self.root, text, parser.blocks, {"format": "html"})


class JSONLoader:
    """把 JSON 数组或对象转换成记录级 blocks。"""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self, path: Path) -> SourceDocument:
        """读取 JSON，并为每条记录保留结构化序列化结果。"""

        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        records = payload if isinstance(payload, list) else [payload]
        blocks = []
        for index, record in enumerate(records):
            rendered = json.dumps(record, ensure_ascii=False, sort_keys=True)
            blocks.append(Block("json_record", rendered, metadata={"record_index": index}))
        return build_source_document(path, self.root, text, blocks, {"format": "json"})


def build_source_document(
    path: Path,
    root: Path,
    raw_text: str,
    blocks: list[Block],
    metadata: dict[str, object],
) -> SourceDocument:
    """统一构建 SourceDocument，避免各 loader 重复拼装元数据。"""

    return SourceDocument(
        doc_id=stable_doc_id(path, root),
        source_uri=path.relative_to(root).as_posix(),
        content_hash=sha256_text(raw_text),
        updated_at=utc_iso_from_mtime(path),
        blocks=blocks,
        metadata=metadata,
    )


class StructureAwareChunker:
    """结构优先的递归 chunker。

    先尽量保持 loader 产出的 block 完整；当 block 太长时，再依次按段落、
    句子、空格和最终字符窗口切分。相邻 chunk 保留 token overlap。
    """

    SEPARATORS = ("\n\n", "\n", "。", ". ", "；", "; ", "，", ", ", " ")

    def __init__(self, max_tokens: int = 420, overlap_tokens: int = 60) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须大于 0")
        if overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens 必须满足 0 <= overlap_tokens < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @property
    def version(self) -> str:
        """返回包含关键参数的 chunker 版本，用于增量更新判断。"""

        return f"structure-recursive-v1-size{self.max_tokens}-overlap{self.overlap_tokens}"

    def split(self, document: SourceDocument) -> list[Chunk]:
        """把结构化文档切成最终 chunks。"""

        chunk_texts: list[tuple[str, tuple[str, ...], dict[str, object]]] = []
        for block in document.blocks:
            pieces = self._split_text(block.text)
            for piece in pieces:
                metadata = {**document.metadata, **block.metadata, "block_kind": block.kind}
                chunk_texts.append((piece, block.title_path, metadata))

        chunks = []
        for index, (text, title_path, metadata) in enumerate(chunk_texts):
            chunk_hash = sha256_text(f"{document.content_hash}:{index}:{text}")
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index}:{chunk_hash[:10]}",
                    doc_id=document.doc_id,
                    source_uri=document.source_uri,
                    title_path=title_path,
                    text=text,
                    chunk_index=index,
                    token_count=len(tokenize(text)),
                    content_hash=chunk_hash,
                    updated_at=document.updated_at,
                    parser_version=PARSER_VERSION,
                    chunker_version=self.version,
                    metadata=metadata,
                )
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """递归切分超长文本，并给相邻窗口补 overlap。"""

        if len(tokenize(text)) <= self.max_tokens:
            return [text.strip()]
        pieces = self._recursive_split(text, 0)
        return self._add_overlap(pieces)

    def _recursive_split(self, text: str, separator_index: int) -> list[str]:
        """依次尝试语义更自然的分隔符，最后回退到 token 窗口。"""

        if len(tokenize(text)) <= self.max_tokens:
            return [text.strip()]
        if separator_index >= len(self.SEPARATORS):
            return self._token_windows(text)

        separator = self.SEPARATORS[separator_index]
        parts = [part.strip() for part in text.split(separator) if part.strip()]
        if len(parts) == 1:
            return self._recursive_split(text, separator_index + 1)

        chunks: list[str] = []
        buffer: list[str] = []
        for part in parts:
            candidate = separator.join(buffer + [part]).strip()
            if buffer and len(tokenize(candidate)) > self.max_tokens:
                merged = separator.join(buffer).strip()
                chunks.extend(self._recursive_split(merged, separator_index + 1))
                buffer = [part]
            else:
                buffer.append(part)
        if buffer:
            chunks.extend(self._recursive_split(separator.join(buffer), separator_index + 1))
        return chunks

    def _token_windows(self, text: str) -> list[str]:
        """字符边界完全不可用时，回退到 token 窗口。"""

        tokens = tokenize(text)
        return [" ".join(tokens[start : start + self.max_tokens]) for start in range(0, len(tokens), self.max_tokens)]

    def _add_overlap(self, pieces: list[str]) -> list[str]:
        """把前一个 chunk 尾部 token 复制到后一个 chunk 开头。"""

        if self.overlap_tokens == 0 or len(pieces) <= 1:
            return pieces

        with_overlap = [pieces[0]]
        for previous, current in zip(pieces, pieces[1:]):
            # overlap 前缀只负责补上下文；当前 chunk 保持原始大小写和标点。
            prefix = TOKEN_RE.findall(previous)[-self.overlap_tokens :]
            with_overlap.append(" ".join(prefix) + " " + current)
        return with_overlap


def tokenize(text: str) -> list[str]:
    """用于 demo 的轻量 tokenization。

    生产系统通常会使用目标 embedding 模型对应的 tokenizer 统计 token。
    """

    return [token.lower() for token in TOKEN_RE.findall(text)]


class IngestionPipeline:
    """串联 loader、chunker、manifest 和 JSONL sink。"""

    def __init__(self, input_dir: Path, output_dir: Path, chunker: StructureAwareChunker) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunker = chunker
        self.manifest_path = output_dir / "manifest.json"
        self.chunks_path = output_dir / "chunks.jsonl"

    def run(self) -> dict[str, object]:
        """执行一次全量扫描 + 增量处理。"""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        previous = self._read_manifest()
        previous_chunks = self._read_chunks_by_source()
        current_manifest: dict[str, dict[str, object]] = {}
        all_chunks: list[Chunk] = []
        stats = {"processed": 0, "unchanged": 0, "deleted": 0, "chunks": 0, "reused_chunks": 0}

        for path in self._scan_files():
            loader = self._select_loader(path)
            document = loader.load(path)
            old = previous.get(document.source_uri)
            unchanged = (
                old
                and old.get("content_hash") == document.content_hash
                and old.get("parser_version") == PARSER_VERSION
                and old.get("chunker_version") == self.chunker.version
            )

            if unchanged:
                stats["unchanged"] += 1
                chunks = previous_chunks.get(document.source_uri, [])
                stats["reused_chunks"] += len(chunks)
            else:
                stats["processed"] += 1
                chunks = self.chunker.split(document)
            all_chunks.extend(chunks)
            current_manifest[document.source_uri] = {
                "doc_id": document.doc_id,
                "content_hash": document.content_hash,
                "updated_at": document.updated_at,
                "chunk_count": len(chunks),
                "parser_version": PARSER_VERSION,
                "chunker_version": self.chunker.version,
            }

        stats["deleted"] = len(set(previous) - set(current_manifest))
        stats["chunks"] = len(all_chunks)
        self._write_chunks(all_chunks)
        self._write_manifest(current_manifest)
        return stats

    def _scan_files(self) -> list[Path]:
        """递归扫描受支持文件，结果排序保证输出稳定。"""

        return sorted(
            path
            for path in self.input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )

    def _select_loader(self, path: Path) -> Loader:
        """按扩展名选择 loader。"""

        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return MarkdownLoader(self.input_dir)
        if suffix in {".html", ".htm"}:
            return HTMLLoader(self.input_dir)
        if suffix == ".json":
            return JSONLoader(self.input_dir)
        return TextLoader(self.input_dir)

    def _read_manifest(self) -> dict[str, dict[str, object]]:
        """读取上一次 manifest，不存在时返回空映射。"""

        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _read_chunks_by_source(self) -> dict[str, list[Chunk]]:
        """读取旧 JSONL 快照，供未变化文档直接复用 chunks。"""

        if not self.chunks_path.exists():
            return {}

        chunks_by_source: dict[str, list[Chunk]] = {}
        for line in self.chunks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["title_path"] = tuple(payload["title_path"])
            chunk = Chunk(**payload)
            chunks_by_source.setdefault(chunk.source_uri, []).append(chunk)
        return chunks_by_source

    def _write_manifest(self, manifest: dict[str, dict[str, object]]) -> None:
        """写入文档级 manifest。"""

        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_chunks(self, chunks: Iterable[Chunk]) -> None:
        """把 chunk 完整快照写成 JSONL，便于检查或导入索引。"""

        lines = [json.dumps(asdict(chunk), ensure_ascii=False, sort_keys=True) for chunk in chunks]
        self.chunks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
