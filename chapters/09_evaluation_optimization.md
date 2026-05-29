# 第 9 章：评估、优化与回归测试

## 分层评估

RAG 评估必须拆层：

- 数据层：知识是否存在、是否过期、是否有权限。
- 检索层：相关证据是否进入 top-k。
- 排序层：相关证据是否排在上下文预算内。
- 上下文层：是否去重、是否冲突、是否顺序合理。
- 生成层：答案是否忠实、完整、有引用。
- 系统层：延迟、成本、缓存、失败率。

## 检索指标

```text
Recall@k = top-k 中相关文档数 / 全部相关文档数
MRR = 1 / 第一个相关文档的排名
nDCG = DCG / 理想 DCG
```

优化策略：

- 先保证 Recall@k。
- 再优化 MRR 和 nDCG。
- 最后看生成质量。

如果证据没被检索到，prompt 再好也只是让模型更会编。

## 生成指标

- Faithfulness：答案 claim 是否被上下文支持。
- Answer relevance：是否回答用户问题。
- Citation precision：引用是否真的支持对应句子。
- Citation recall：关键结论是否都有引用。
- Abstention quality：资料不足时是否拒答。

ARES、RAGBench 等工作都强调把 context relevance、answer faithfulness、answer relevance 拆开看。

## Demo

运行：

```bash
python demos/09_eval_regression.py
```

这个 demo 对 hybrid + rerank 管线跑一个小评估集，输出 `Recall@3` 和 `MRR`。真实项目里应把问题集保存为 JSONL，并在每次改索引、prompt、模型、chunker 后跑回归。

## 优化理论

优先级：

1. 语料是否包含答案。
2. chunk 是否保留语义。
3. relevant chunk 是否召回。
4. relevant chunk 是否排得足够前。
5. 上下文是否压缩和排序合理。
6. prompt 是否要求基于证据。
7. 模型是否具备必要推理能力。

## 参考

- ARES, 2023: https://arxiv.org/abs/2311.09476
- RAGBench, 2024: https://arxiv.org/abs/2407.11005
- BEIR, 2021: https://arxiv.org/abs/2104.08663

