# 第 7 章：Multi-hop、RAPTOR 与层级检索

## Multi-hop RAG

Multi-hop RAG 用于需要多个证据组合的问题。

流程：

```text
question
-> decompose into sub-questions
-> retrieve for each sub-question
-> merge evidence
-> resolve conflicts
-> synthesize final answer
```

关键不是“多查几次”，而是每一跳都有明确目的和停止条件。

## RAPTOR 与层级摘要

RAPTOR 的思路是把文档 chunk 聚类，再递归生成摘要树。检索时可以在不同层级上找证据：

- 底层 chunk：适合细节事实。
- 中层摘要：适合章节级主题。
- 高层摘要：适合全局问题。

这解决了朴素 top-k RAG 偏向局部片段的问题。

## Demo

运行：

```bash
python demos/07_multihop_rag.py
```

demo 会把“对比 sparse 和 dense retrieval”拆成多个子问题，分别检索，再合并证据生成答案。

## 设计要点

- 子问题不要无限递归。
- 每一跳保存 trace。
- 合并证据时去重。
- 有冲突时显式报告。
- 全局问题可以先检索摘要，再回到底层 chunk 找引用。

## 参考

- RAPTOR, 2024: https://arxiv.org/abs/2401.18059

