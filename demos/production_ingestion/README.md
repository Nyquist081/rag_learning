# 生产化文档加载与 Chunking Demo

前三章的基础 demo 用于解释概念。这个目录展示更接近生产环境的 ingestion 管线。

## 运行

```bash
python demos/production_ingestion/run_pipeline.py
```

默认读取：

```text
demos/production_ingestion/sample_docs/
```

默认输出：

```text
/tmp/rag-production-ingestion/chunks.jsonl
/tmp/rag-production-ingestion/manifest.json
```

再次运行时，`manifest.json` 会识别未变化文档：

```bash
python demos/production_ingestion/run_pipeline.py
```

## 管线结构

```text
scan files
-> choose loader by suffix
-> parse blocks
-> structure-aware recursive chunking
-> add overlap
-> attach metadata
-> compute hashes
-> write chunks.jsonl
-> write manifest.json
```

## 支持格式

- Markdown：保留标题层级、代码块、表格。
- HTML：保留标题、段落、列表、代码。
- JSON：按记录切分。
- TXT：按段落加载。

PDF、Word、PPT、Excel 在生产系统里通常交给专用解析器，例如 Unstructured、Docling、LlamaParse。这个 demo 刻意只使用 Python 标准库，方便直接运行和阅读。

## 输出字段

每个 chunk 包含：

- `chunk_id`
- `doc_id`
- `source_uri`
- `title_path`
- `text`
- `chunk_index`
- `token_count`
- `content_hash`
- `updated_at`
- `parser_version`
- `chunker_version`
- `metadata`

这些字段用于引用、权限过滤、增量更新、索引重建、日志回放和线上排障。

## 生产环境如何替换

| Demo 组件 | 生产环境替换 |
| --- | --- |
| 标准库 Markdown/HTML loader | Unstructured、Docling、LlamaParse、自研 parser |
| 本地目录扫描 | S3、对象存储、数据库、消息队列、CDC |
| 标准库 tokenization | embedding 模型对应 tokenizer |
| JSONL 输出 | Elasticsearch、OpenSearch、向量数据库、docstore |
| 单进程执行 | worker pool、任务队列、批量 embedding |
| manifest 文件 | PostgreSQL、Redis、对象存储 metadata 表 |

## 参考资料

- LlamaIndex Ingestion Pipeline: https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion_pipeline/
- LangChain Text Splitters: https://docs.langchain.com/oss/python/integrations/splitters/index
- Unstructured Partitioning: https://docs.unstructured.io/concepts/partitioning
- Unstructured Chunking: https://docs.unstructured.io/open-source/core-functionality/chunking
- Elasticsearch `semantic_text`: https://www.elastic.co/docs/reference/elasticsearch/mapping-reference/semantic-text-reference

