# 第 3 章检索系统设计说明

日期：2026-05-30

## 目标

将第 3 章 `Sparse、Dense 与 Hybrid Retrieval` 扩写成完整教程章节。深度和写法对齐第 1、2 章：先讲理论，再讲实现原理、工程取舍、失败模式、demo 对照和实践检查清单。

读完本章后，读者应该能做到：

- 解释检索模块在 RAG 系统里负责什么。
- 区分 sparse retrieval、dense retrieval 和 hybrid retrieval。
- 从工程角度理解 BM25 公式。
- 理解 embedding 检索、向量相似度，以及为什么“语义相似”不等于“能支持答案”。
- 解释为什么生产 RAG 常用 hybrid retrieval。
- 理解 RRF 和加权融合的使用方式与取舍。
- 在责怪生成模型之前，先诊断检索链路的问题。
- 读懂并解释 `demos/03_sparse_dense_hybrid.py` 以及 `demos/rag_core.py` 中相关函数。

## 范围

本设计只覆盖第 3 章及其现有 demo 的解释。

范围内：

- 重写 `chapters/03_retrieval_sparse_dense_hybrid.md`。
- 保持当前 demo 代码不变，除非出现很小的文档驱动型调整。
- 引用当前公开研究和成熟检索概念。
- 保持第 3 章在课程中的位置：承接第 2 章索引与 chunk，为第 4 章 query rewrite 和第 5 章 reranking 铺路。

范围外：

- 不深入实现 reranking。第 5 章负责 reranking。
- 不展开 query rewriting 和 HyDE。第 4 章负责这些内容。
- 不重复讲完整 ANN 内部原理。第 2 章已经介绍索引；第 3 章只说明向量搜索在检索链路中的位置。
- 不展开生产监控和安全治理。后续章节覆盖这些内容。

## 推荐写法

采用“分层发动机舱式”结构：从检索问题的本质开始，依次加入 sparse retrieval、dense retrieval、hybrid retrieval、调参、诊断和 demo 对照。

这样能平衡理论和工程实践，避免章节变成纯数学笔记，也避免变成只讲经验、不讲算法的生产手册。

## 章节结构

扩写后的章节应包含以下部分：

1. **检索在 RAG 中的角色**
   - 检索是候选证据选择，不是最终回答。
   - retriever 的目标是在上下文预算内提高召回。
   - 检索质量会限制最终答案质量上限。

2. **形式化检索问题**
   - 定义 query `q`、语料库 `D`、候选文档 `d`、打分函数 `s(q, d)` 和 `top-k`。
   - 解释 RAG 中 recall 和 precision 的取舍。
   - 区分 candidate pool 和 final context。

3. **Sparse retrieval**
   - 解释倒排索引。
   - 解释 TF、IDF、文档长度归一。
   - 给出 BM25 公式和直觉。
   - 解释 `k1` 和 `b`。
   - 说明优势：精确标识符、产品名、错误码、代码符号。
   - 说明弱点：同义改写、词汇不匹配、多语言语义匹配。

4. **Dense retrieval**
   - 解释 embedding 模型和向量空间。
   - 解释 cosine similarity、dot product 和归一化。
   - 解释语义相似和同义表达匹配。
   - 说明 dense retrieval 失败模式：漏掉精确 token、错误语义邻居、embedding 模型不匹配、领域漂移。

5. **ANN 和向量搜索在本章中的位置**
   - 简要解释为什么暴力向量搜索成本高。
   - 说明 HNSW、IVF、PQ、向量数据库在检索管线里的位置。
   - 不重复第 2 章的索引细节。

6. **Hybrid retrieval**
   - 解释为什么生产 RAG 常用多路召回。
   - 说明候选集合并、去重、分数归一化、加权融合和 RRF。
   - 给出 RRF 公式，并解释为什么它在分数不可比时很稳健。
   - 说明加权融合什么时候有用、什么时候有风险。

7. **元数据和权限过滤**
   - 解释 pre-filter 和 post-filter。
   - 说明为什么 top-k 后再过滤会损害召回，甚至泄漏 trace。
   - 覆盖时间、文档类型、租户、语言、ACL 等过滤条件。

8. **失败模式和诊断**
   - top-k 中没有相关证据。
   - 相关证据被召回但排名太低。
   - dense retrieval 漏掉标识符。
   - BM25 漏掉同义表达。
   - hybrid 返回重复结果或噪声近邻。
   - top-k 太小会漏证据，top-k 太大会污染生成上下文。

9. **调参指南**
   - `top_k`。
   - rerank 前 candidate pool 大小。
   - BM25 `k1` 和 `b`。
   - embedding 模型选择。
   - 相似度函数和归一化。
   - RRF 常数。
   - 融合权重。

10. **Demo 对照讲解**
    - 将 `BM25Index(CORPUS).search(...)` 对应到 sparse retrieval。
    - 将 `dense_style_search(...)` 对应到简化版 dense retrieval。
    - 将 `reciprocal_rank_fusion(...)` 对应到 hybrid retrieval。
    - 解释为什么 demo 用词项扩展模拟 dense retrieval，而不是调用真实 embedding 模型。
    - 说明后续如何把玩具版 dense retriever 替换成真实 embedding 模型。

11. **实践检查清单**
    - 应该记录哪些 trace。
    - 应该观察哪些指标。
    - 新语料库如何选择 sparse、dense 或 hybrid。
    - 在调生成模型之前，应该先问哪些检索问题。

12. **参考资料**
    - BEIR。
    - BM25 / 概率相关性框架。
    - Dense Passage Retrieval。
    - ColBERT 或 late interaction 作为桥接概念。
    - RAG Survey。
    - 如果直接有用，补充近期 retrieval / hybrid RAG 资料。

## 算法深度

BM25 需要包含公式：

```text
score(q, d) = sum IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

章节需要解释：

- `f(t,d)` 表示词项在文档中的出现频率。
- `IDF(t)` 表示词项稀有程度。
- `|d| / avgdl` 是文档长度归一。
- `k1` 控制词频饱和速度。
- `b` 控制文档长度对分数的影响程度。

Dense retrieval 需要包含：

```text
v_q = embed(q)
v_d = embed(d)
score = cosine(v_q, v_d)
```

Hybrid retrieval 需要包含：

```text
RRF(d) = sum 1 / (k + rank_i(d))
```

数学内容服务于理解，不把章节写成证明导向的论文笔记。

## Demo 要求

现有 demo 应保持简单、无第三方依赖：

```bash
python demos/03_sparse_dense_hybrid.py
```

章节要明确说明：`dense_style_search` 不是真实 embedding retriever。它使用词项扩展和余弦相似度，是为了在课程中不依赖外部模型或向量数据库，也能演示 dense-style 行为。

## 验收标准

- 第 3 章扩写为自包含教程。
- 同时解释算法原理和工程取舍。
- 明确区分 retrieval、reranking 和 generation。
- 包含 BM25、cosine similarity、RRF 的公式。
- demo 讲解和当前代码一致。
- 不重复第 2 章对索引结构的深讲。
- 不把第 4 章 query rewriting 或第 5 章 reranking 提前塞进本章。
- 现有 demo 脚本仍可运行。

## 验证方式

实现后运行：

```bash
python demos/03_sparse_dense_hybrid.py
python -m py_compile demos/*.py practice/toy_rag.py
```

检查 git diff，确认只修改预期文件。

