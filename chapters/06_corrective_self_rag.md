# 第 6 章：Corrective RAG 与 Self-RAG

## Corrective RAG

Corrective RAG 的核心是：检索结果不可靠时，不要直接生成答案。

典型流程：

```text
retrieve
-> grade evidence
-> if weak: rewrite/search again
-> if still weak: refuse or ask clarification
-> answer with evidence
```

证据可以分级：

- `strong`：直接支持答案。
- `partial`：相关但不完整。
- `weak`：主题相邻但无法回答。
- `conflicting`：证据冲突。
- `missing`：没有证据。

## Self-RAG

Self-RAG 让模型学习或模拟几个自反思判断：

- 是否需要检索。
- 检索结果是否相关。
- 生成内容是否被支持。
- 答案是否有用。

工程上不一定要训练模型，可以先用小模型、规则、LLM judge 或人工标注实现这些判断器。

## Demo

运行：

```bash
python demos/06_corrective_rag.py
```

demo 首先检索并给证据打分。如果证据不足，会改写 query 再检索，模拟 Corrective RAG 的补救流程。

## 什么时候上 CRAG/Self-RAG

适合：

- 答案准确性要求高。
- 语料噪声大。
- 用户问题复杂。
- 召回经常主题相关但证据不足。

不适合：

- 极低延迟场景。
- 简单 FAQ 且召回已经稳定。
- 没有评估集，无法判断纠错是否真的有效。

## 参考

- Self-RAG, 2023: https://openreview.net/forum?id=jbNjgmE0OP
- Corrective RAG, 2024: https://arxiv.org/abs/2401.15884

