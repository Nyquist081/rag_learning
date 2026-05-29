# 第 4 章：Query Rewrite、Multi-query 与 HyDE

## 为什么需要查询改写

用户问题经常不适合直接检索：

- 太短。
- 有指代。
- 用业务黑话描述技术问题。
- 问题是多跳的。
- 问题里没有目标文档使用的关键词。

Query rewrite 的目标是提高召回，而不是替用户回答。

## 常见方法

### Rewrite

把用户问题改写成更完整、更适合检索的查询。

### Multi-query

生成多个角度的查询，分别召回后去重融合。

### HyDE

HyDE 先生成一个“假想答案/假想文档”，再用这个假想文档去检索真实文档。它对零样本 dense retrieval 有帮助，但风险是把模型幻觉引入检索。

### Step-back

先抽象出上位概念，再检索背景知识，适合用户问题太具体但缺上下文的场景。

## Demo

运行：

```bash
python demos/04_query_rewrite_hyde.py
```

这个 demo 把 `bad retrieval` 改写成包含 `weak evidence`、`corrective rag`、`rewrite query` 等检索词，模拟 HyDE/query rewrite 对召回的帮助。

## 质量控制

- 保存原始 query 和改写 query。
- 改写不能改变用户意图。
- 多查询要去重。
- 改写失败时回退到原始 query。
- 对高风险场景，改写结果需要可解释或可审计。

## 参考

- HyDE, 2022: https://arxiv.org/abs/2212.10496

