# RAG Learning

这是一套截至 2026-05 的 RAG 系统学习教程，覆盖从朴素 RAG 到高阶 RAG、评估、优化和生产安全。教程按章节推进，每章都有对应 demo，代码从最小系统逐步演进。

## 学习方式

建议顺序：

1. 先读章节文档。
2. 运行对应 demo。
3. 改 `demos/rag_core.py` 里的 `CORPUS`，替换成自己的资料。
4. 每次改检索、重排或 prompt，都运行评估 demo。

所有 demo 都是无依赖 Python：

```bash
python demos/01_naive_rag.py
python demos/03_sparse_dense_hybrid.py
python demos/09_eval_regression.py
```

## 课程目录

| 章节 | 主题 | Demo |
| --- | --- | --- |
| [第 1 章](chapters/01_rag_foundation.md) | RAG 基础与系统边界 | [01_naive_rag.py](demos/01_naive_rag.py) |
| [第 2 章](chapters/02_data_chunking_indexing.md) | 数据、切分与索引 | [02_chunking_metadata.py](demos/02_chunking_metadata.py) |
| [第 3 章](chapters/03_retrieval_sparse_dense_hybrid.md) | Sparse、Dense、Hybrid Retrieval | [03_sparse_dense_hybrid.py](demos/03_sparse_dense_hybrid.py) |
| [第 4 章](chapters/04_query_rewrite_hyde.md) | Query Rewrite、Multi-query、HyDE | [04_query_rewrite_hyde.py](demos/04_query_rewrite_hyde.py) |
| [第 5 章](chapters/05_reranking_grounding.md) | Reranking、上下文构造、Grounding | [05_rerank_grounding.py](demos/05_rerank_grounding.py) |
| [第 6 章](chapters/06_corrective_self_rag.md) | Corrective RAG 与 Self-RAG | [06_corrective_rag.py](demos/06_corrective_rag.py) |
| [第 7 章](chapters/07_multihop_hierarchical_rag.md) | Multi-hop、RAPTOR、层级检索 | [07_multihop_rag.py](demos/07_multihop_rag.py) |
| [第 8 章](chapters/08_graph_rag.md) | GraphRAG、LightRAG、图结构检索 | [08_graph_rag.py](demos/08_graph_rag.py) |
| [第 9 章](chapters/09_evaluation_optimization.md) | 评估、优化与回归测试 | [09_eval_regression.py](demos/09_eval_regression.py) |
| [第 10 章](chapters/10_production_security.md) | 生产化、安全与持续迭代 | [10_security_filtering.py](demos/10_security_filtering.py) |

## Demo 演进路线

```text
01 naive BM25 RAG
-> 02 metadata and traceability
-> 03 sparse + dense-style + RRF hybrid
-> 04 query rewrite / HyDE-like expansion
-> 05 support-aware reranking and grounded answer
-> 06 corrective retrieval
-> 07 multi-hop decomposition
-> 08 graph-expanded retrieval
-> 09 evaluation regression
-> 10 security filtering
```

核心共享代码在 [demos/rag_core.py](demos/rag_core.py)。

## 生产化 Ingestion Demo

前三章的 toy demo 用来理解算法。要观察更接近生产环境的文档 loading 和 chunking 管线，运行：

```bash
python demos/production_ingestion/run_pipeline.py
```

详细说明见 [demos/production_ingestion/README.md](demos/production_ingestion/README.md)。

## 技术地图

基础能力：

- 文档清洗、切分、元数据。
- BM25 和 sparse retrieval。
- embedding 和 dense retrieval。
- hybrid retrieval 和 RRF。
- reranking。
- grounded generation 和 citations。

高阶能力：

- query rewriting。
- HyDE。
- Corrective RAG。
- Self-RAG。
- Multi-hop RAG。
- RAPTOR 和层级摘要检索。
- GraphRAG、LightRAG。
- RAG evaluation。
- RAG security。

## 资料来源

主资料索引见 [research_notes/sources.md](research_notes/sources.md)。代表性来源包括：

- RAG, Lewis et al., 2020: https://arxiv.org/abs/2005.11401
- RAG Survey, Gao et al., 2023: https://arxiv.org/abs/2312.10997
- HyDE, 2022: https://arxiv.org/abs/2212.10496
- Self-RAG, 2023: https://openreview.net/forum?id=jbNjgmE0OP
- Corrective RAG, 2024: https://arxiv.org/abs/2401.15884
- RAPTOR, 2024: https://arxiv.org/abs/2401.18059
- Microsoft GraphRAG: https://www.microsoft.com/en-us/research/project/graphrag/
- LightRAG, 2024: https://arxiv.org/abs/2410.05779
- ARES, 2023: https://arxiv.org/abs/2311.09476
- RAGBench, 2024: https://arxiv.org/abs/2407.11005
- OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications

## 旧版材料

- [advanced_rag_checklist.md](advanced_rag_checklist.md)
- [practice/toy_rag.py](practice/toy_rag.py)
