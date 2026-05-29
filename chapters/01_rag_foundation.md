# 第 1 章：RAG Foundation

本章目标是把 RAG 的“为什么”和“最小实现原理”讲清楚。读完后，你应该能回答：

- RAG 到底解决什么问题？
- RAG 和微调、长上下文、普通搜索有什么区别？
- 一个最小 RAG 系统由哪些模块组成？
- 检索结果为什么能影响生成结果？
- 什么时候 RAG 有用，什么时候它反而不是最优解？
- 第一个 demo 里的每一步对应真实系统里的哪个组件？

## 1. RAG 要解决的问题

大语言模型有两类知识来源：

- 参数化知识：模型训练后存在权重里的知识。
- 非参数化知识：外部文档、数据库、网页、代码库、日志、工单、知识图谱等可检索知识。

只依赖参数化知识会遇到几个问题：

- 知识会过期。
- 私有数据没有进入模型训练。
- 模型不知道自己不知道，容易生成看似合理但无证据的答案。
- 企业数据有权限边界，不能靠模型记忆控制访问。
- 事实更新不应该每次都靠重新训练或微调。

RAG 的思路是：不要要求模型把所有知识都记在参数里，而是在推理时先从外部知识库取出和问题相关的证据，再让模型基于这些证据回答。

可以把 RAG 理解成：

```text
用户问题
-> 检索系统找到相关证据
-> 把问题和证据一起交给生成模型
-> 模型基于证据生成答案
```

这件事的价值不只是“答案更准”，更重要的是知识变成了可更新、可审计、可权限控制、可评估的外部系统。

## 2. 原始 RAG 的核心思想

Lewis et al. 2020 的 RAG 论文把生成任务拆成两个来源：

- 语言模型本身的参数化记忆。
- 通过检索访问的非参数化记忆。

论文中的模型会先为输入问题检索若干文档，再把检索文档作为隐变量参与生成。工程上不一定要完全复现论文里的端到端训练形式，但这个思想非常关键：答案不是只由 `question` 决定，而是由 `question + retrieved evidence` 共同决定。

朴素工程近似可以写成：

```text
answer = generator(question, retrieve(question, corpus))
```

稍微形式化一点：

```text
z = top_k(retriever(q, D))
y = generator(q, z)
```

其中：

- `D` 是外部知识库。
- `q` 是用户问题。
- `z` 是检索到的证据集合。
- `y` 是最终答案。

这解释了 RAG 的第一个基本原则：生成质量上限受检索质量约束。证据没被召回，模型只能依赖自己的参数化记忆或猜测。

## 3. RAG 和几个相邻概念的区别

### RAG vs 普通搜索

普通搜索返回文档列表，用户自己阅读、判断和整合。

RAG 多了一步生成：

```text
search -> documents
rag -> documents -> synthesized answer
```

因此 RAG 不只是信息检索系统，也不是纯生成系统。它是“检索 + 证据选择 + 上下文构造 + 生成”的组合系统。

### RAG vs Prompt Stuffing

Prompt stuffing 是把一堆材料直接塞进 prompt。它的问题是：

- 成本高。
- 噪声多。
- 证据顺序和位置会影响模型使用效果。
- 权限和审计不清晰。
- 文档规模一大就不可持续。

RAG 的关键是先选择证据，再生成，而不是把所有东西都塞给模型。

### RAG vs 微调

微调适合改变模型行为、输出风格、任务格式或领域能力。

RAG 适合接入会变化的事实知识、私有知识和需要引用的知识。

粗略判断：

| 需求 | 更适合 |
| --- | --- |
| 回答最新公司制度 | RAG |
| 回答内部产品文档 | RAG |
| 固定输出 JSON 格式 | 微调或指令优化 |
| 学会特定写作风格 | 微调或 few-shot |
| 更新一条价格、政策、配置 | RAG |
| 大量领域术语理解能力不足 | RAG + 可能微调 |

实践里经常是组合：用 RAG 提供知识，用 prompt/微调约束模型行为。

### RAG vs 长上下文

长上下文模型能装更多内容，但它没有消除 RAG：

- 不可能每次请求都塞入完整企业知识库。
- 长上下文成本和延迟更高。
- 数据权限仍然要在上下文构造前处理。
- 相关证据放在长上下文中间时，模型使用效果可能下降。
- 检索 trace 和引用仍然是生产系统需要的审计材料。

`Lost in the Middle` 的结论提醒我们：更长的上下文不等于模型能稳定利用任意位置的信息。RAG 的价值是把更少、更相关、更可控的证据放进上下文。

## 4. 一个 RAG 系统的两条链路

RAG 有离线链路和在线链路。

### 离线链路：构建知识库

```text
原始数据
-> 加载
-> 清洗
-> 切分
-> 元数据绑定
-> 建索引
```

每一步都很重要：

- 加载：从 Markdown、PDF、网页、数据库、代码仓库等来源读入。
- 清洗：去掉导航、广告、乱码、重复内容，但保留标题、表格、代码块等结构。
- 切分：把长文档切成可检索的 chunk。
- 元数据绑定：保留 source、标题路径、更新时间、权限、版本。
- 建索引：建立 BM25、向量索引、图索引或混合索引。

这里最容易犯的错误是把文档处理当成“前置杂活”。实际上，数据层决定了 RAG 的质量底座。

### 在线链路：回答用户问题

```text
用户问题
-> 查询理解
-> 检索
-> 重排
-> 上下文构造
-> 生成
-> 引用与校验
-> 记录 trace
```

朴素 RAG 可以先省略查询改写、重排和校验，但生产系统通常不能省略 trace。你至少要知道：

- 用户问了什么。
- 用什么 query 检索。
- 召回了哪些 chunk。
- 每个 chunk 的分数和来源是什么。
- 最终 prompt 放入了哪些证据。
- 答案引用了哪些证据。

没有 trace，就无法系统优化。

## 5. 最小 RAG 的三个核心组件

### 组件 1：知识库

知识库不是一段文本，而是一组带元数据的文档片段。

最小结构：

```python
Document(
    id="d01",
    title="RAG definition",
    text="RAG combines a retriever with a generator...",
    source="notes/rag_basics.md",
    tags=("basics",),
)
```

为什么要有 `id` 和 `source`？

- `id` 用于评估、去重、引用、日志回放。
- `source` 用于用户检查答案依据。
- `tags` 或 ACL 用于过滤和权限控制。

如果没有这些元数据，系统即使答对，也很难证明为什么答对。

### 组件 2：检索器

检索器的任务是从知识库中挑出和问题相关的候选证据。

第 1 章 demo 用 BM25 作为最小检索器。BM25 属于 sparse retrieval，它基于词项匹配和词频统计。虽然它不是最新技术，但非常适合教学和很多生产场景的 baseline，因为：

- 不需要 embedding 模型。
- 不需要向量数据库。
- 对关键词、错误码、专有名词很有效。
- 排序逻辑相对可解释。

BM25 的直觉：

- query 里的词在文档中出现越多，相关性通常越高。
- 越稀有的词越重要。
- 文档太长时要做长度归一，避免长文档天然占优。

这对应三个信号：

```text
term frequency: 词在文档中出现多少次
inverse document frequency: 词在整个语料中有多稀有
length normalization: 文档长度修正
```

### 组件 3：生成器

生成器接收问题和证据，输出答案。

最小 RAG demo 没有调用真实 LLM，而是用模板模拟 grounded answer：

```text
问题：...
回答：
1. evidence chunk text [chunk_id]
引用：source#chunk_id
```

这样做是为了先把 RAG 的证据链讲清楚。真实系统里，模板会换成 LLM prompt，但原则不变：

- 答案必须基于证据。
- 证据不足时要允许拒答。
- 关键结论要能追溯到 chunk。

## 6. 检索为什么会改变生成

LLM 生成答案时，本质上是在给定上下文中预测接下来的 token。如果上下文只有用户问题，模型只能依赖参数化记忆和推理能力。

加入证据后，上下文变成：

```text
system instruction
user question
retrieved evidence
answer requirements
```

这会改变模型的条件输入。工程上我们希望模型从：

```text
p(answer | question)
```

变成：

```text
p(answer | question, evidence)
```

这就是 RAG 能减少幻觉的理论直觉：它把回答条件从“只看问题”变成“看问题和外部证据”。

但注意，RAG 不是自动保证正确。它会引入新的失败模式：

- 证据没召回。
- 召回了错误证据。
- 召回证据主题相关但不能支持答案。
- 上下文太长，模型忽略关键证据。
- 模型没有遵守“只基于证据回答”的要求。
- 引用和答案 claim 对不上。

所以后续章节会逐步加入 hybrid retrieval、query rewrite、rerank、corrective RAG、evaluation 和 security。

## 7. Naive RAG、Advanced RAG、Modular RAG

2023 年之后的 RAG 综述常把 RAG 演进分成三类。

### Naive RAG

典型流程：

```text
index -> retrieve -> generate
```

优点：

- 简单。
- 容易实现。
- 适合作为 baseline。

缺点：

- 检索 query 原样使用，容易召回不足。
- top-k 直接塞给模型，噪声大。
- 没有证据质量判断。
- 没有系统性评估和纠错。

第 1 章 demo 就是 Naive RAG。

### Advanced RAG

在 Naive RAG 上加入优化：

- 更好的 chunking。
- query rewrite。
- hybrid retrieval。
- reranking。
- context compression。
- citation grounding。
- answer verification。

它的目标是让召回更准、上下文更干净、答案更忠实。

### Modular RAG

把 RAG 拆成可组合模块：

- query router。
- 多检索器。
- 多索引。
- tool calling。
- graph retrieval。
- memory。
- evaluator。
- security guard。

Modular RAG 更适合复杂生产系统，但也更需要 trace、测试和治理。

## 8. 第一个 demo 如何对应真实系统

运行：

```bash
python demos/01_naive_rag.py
```

代码：

```python
from rag_core import BM25Index, CORPUS, grounded_answer, print_ranking

question = "What is RAG with a retriever, generator, external knowledge base, and evidence?"
index = BM25Index(CORPUS)
evidence = index.search(question, k=3)

print_ranking("Naive BM25 retrieval:", evidence)
print()
print(grounded_answer(question, evidence))
```

逐行理解：

| 代码 | RAG 概念 |
| --- | --- |
| `CORPUS` | 知识库 |
| `BM25Index(CORPUS)` | 建立检索索引 |
| `question` | 用户问题 |
| `index.search(question, k=3)` | 召回 top-k 证据 |
| `print_ranking(...)` | 检索 trace |
| `grounded_answer(...)` | 基于证据生成答案 |

你运行后应该看到 `d01 RAG definition` 排在前面。它包含 retriever、generator、external knowledge base、evidence 等关键词，因此 BM25 会给它较高分数。

这也暴露了 BM25 的特点：如果用户换一种完全不同的说法，BM25 可能找不到同一段证据。第 3 章会引入 dense-style retrieval 和 hybrid retrieval 来处理这个问题。

## 9. 判断 RAG 是否适合你的问题

适合 RAG：

- 答案依赖外部知识库。
- 知识经常更新。
- 需要引用来源。
- 需要权限控制。
- 需要企业内部私有知识。
- 需要可审计的回答链路。

不一定适合 RAG：

- 问题主要是创意写作。
- 问题主要是通用推理，不依赖外部资料。
- 数据质量极差且无法清洗。
- 需要严格计算，应该调用数据库或计算工具。
- 任务本质是分类、抽取、格式转换，简单 prompt 或微调就够。

一个实用判断：

```text
如果答案必须来自某个外部资料集合，并且你希望能指出来源，优先考虑 RAG。
```

## 10. RAG 的核心质量原则

### 原则 1：先评估检索，再评估生成

如果正确证据没进 top-k，生成阶段很难稳定答对。

### 原则 2：证据要可引用

没有 source 的上下文只是 prompt 材料，不是可审计证据。

### 原则 3：相关不等于支持

一个 chunk 可能和问题主题相关，但不能直接支持答案。后续 reranking 会重点处理这个问题。

### 原则 4：资料不足时要拒答

RAG 的目标不是“永远回答”，而是“在有证据时回答，在证据不足时诚实说明”。

### 原则 5：RAG 是系统，不是单个模型调用

RAG 的质量来自多个模块共同作用：

```text
data quality
chunking
indexing
retrieval
reranking
context construction
generation
evaluation
monitoring
security
```

只调 prompt 通常解决不了数据和检索问题。

## 11. 本章检查清单

学完本章，你应该能做到：

- 用自己的话解释 RAG 的参数化知识和非参数化知识。
- 画出离线链路和在线链路。
- 解释为什么最小 RAG 至少需要知识库、检索器、生成器。
- 说明 RAG 和微调、长上下文、普通搜索的区别。
- 运行 `demos/01_naive_rag.py` 并解释输出。
- 说出 Naive RAG 的主要缺陷。
- 知道下一步为什么要学习 chunking、hybrid retrieval 和 reranking。

## 参考资料

- Lewis et al., 2020, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: https://arxiv.org/abs/2005.11401
- Gao et al., 2023, Retrieval-Augmented Generation for Large Language Models: A Survey: https://arxiv.org/abs/2312.10997
- Lost in the Middle, 2023: https://arxiv.org/abs/2307.03172

