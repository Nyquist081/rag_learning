# Demo 说明

所有脚本都可以从仓库根目录直接运行：

```bash
python demos/01_naive_rag.py
```

推荐完整跑一遍：

```bash
for file in demos/[0-9][0-9]_*.py; do
  echo "== $file =="
  python "$file"
done
```

## 文件说明

- `rag_core.py`：共享语料、检索器、融合、重排、纠错、评估函数。
- `01_naive_rag.py`：朴素 BM25 RAG。
- `02_chunking_metadata.py`：chunk 元数据和可追溯性。
- `03_sparse_dense_hybrid.py`：BM25、dense-style 和 RRF。
- `04_query_rewrite_hyde.py`：query rewrite 和 HyDE-like expansion。
- `05_rerank_grounding.py`：support-aware rerank 和带引用答案。
- `06_corrective_rag.py`：证据分级和补检索。
- `07_multihop_rag.py`：问题拆解和多跳证据合并。
- `08_graph_rag.py`：图邻域扩展检索。
- `09_eval_regression.py`：Recall@k 和 MRR 回归测试。
- `10_security_filtering.py`：不可信文档过滤。

