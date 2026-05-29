# 第 3 章：Sparse、Dense 与 Hybrid Retrieval

## Sparse Retrieval

Sparse retrieval 代表是 BM25。它依赖词项匹配，适合：

- 错误码。
- 函数名。
- SKU、订单号、配置项。
- 专有名词。
- 用户明确给出的关键词。

弱点是同义改写和语义泛化能力有限。

## Dense Retrieval

Dense retrieval 用 embedding 把 query 和 chunk 映射到向量空间，适合：

- 语义相似。
- 同义表达。
- 跨语言或弱关键词问题。
- 概念性问答。

弱点是可能错过精确标识符，也可能召回“看起来像但不支持答案”的 chunk。

## Hybrid Retrieval

Hybrid retrieval 把 sparse 和 dense 的候选集合合并。常见方式：

- 分数归一化后加权。
- Reciprocal Rank Fusion。
- 学习排序。

RRF 公式：

```text
score(d) = sum(1 / (k + rank_i(d)))
```

RRF 的优点是不要求不同检索器分数可比，只依赖排名。

## Demo

运行：

```bash
python demos/03_sparse_dense_hybrid.py
```

你会看到：

- BM25 更偏关键词。
- demo 里的 dense-style 检索通过同义扩展模拟语义召回。
- hybrid 用 RRF 合并两个排名。

## 优化顺序

1. 先看 relevant chunk 是否在任一召回器 top-k 中。
2. 如果 sparse 能找到、dense 找不到，说明问题偏精确匹配。
3. 如果 dense 能找到、sparse 找不到，说明需要语义召回或 query rewrite。
4. 如果都有但排名低，进入 rerank。

## 参考

- BEIR, 2021: https://arxiv.org/abs/2104.08663
- Gao et al., 2023 Survey: https://arxiv.org/abs/2312.10997

