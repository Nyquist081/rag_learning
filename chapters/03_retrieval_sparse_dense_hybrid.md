# 第 3 章：Sparse、Dense 与 Hybrid Retrieval

本章目标是把 RAG 检索系统的“发动机舱”讲清楚。读完后，你应该能回答：

- 检索在 RAG 中到底负责什么？
- 为什么 BM25 这种老算法仍然是很强的 baseline？
- dense retrieval 为什么能处理同义表达，又为什么会漏掉精确标识符？
- hybrid retrieval 为什么通常比单一路径更稳？
- RRF 为什么适合融合 BM25 和 embedding 检索结果？
- 遇到 RAG 答错时，如何先诊断检索问题，而不是直接调 prompt 或换模型？

## 1. 检索在 RAG 中的角色

RAG 的在线链路可以简化为：

```text
question
-> retrieve candidate evidence
-> build context
-> generate answer
```

检索不是回答问题。检索的职责是：在知识库中找到一批“可能支持答案”的候选证据。

这句话里有两个关键词。

第一，`可能`。检索阶段通常追求召回，不要求最终判断。它要尽量避免漏掉正确证据。

第二，`支持答案`。RAG 不是普通搜索。普通搜索只要求文档主题相关，RAG 需要证据能支撑生成答案。主题相关但不能回答问题的 chunk，会在后续生成阶段制造噪声。

可以把检索看成一个候选池构造问题：

```text
corpus D + query q
-> retriever
-> candidate set C
```

后续 reranker、context builder、generator 都依赖这个候选池。如果正确证据没有进入候选池，后面的模型再强也很难稳定答对。

## 2. 形式化检索问题

定义：

- `D = {d1, d2, ..., dn}`：知识库中的 chunk 集合。
- `q`：用户问题或改写后的检索 query。
- `s(q, d)`：query 和文档 chunk 的相关性分数。
- `top_k(q, D)`：按分数排序后取前 `k` 个 chunk。

检索器做的事情是：

```text
ranked_docs = sort_by_score(D, score=s(q, d))
candidates = ranked_docs[:k]
```

RAG 中有两个不同的 `k`：

- retrieval top-k：检索阶段取多少候选。
- context top-k：最终放进 prompt 的证据数量。

真实系统常会这样做：

```text
retrieve top 50
-> rerank top 50
-> context uses top 5
```

为什么不直接 retrieve top 5？

因为第一阶段检索通常是粗召回。它可能把真正有用的证据排在第 12、第 20。先取更大的 candidate pool，再重排，可以降低漏召回风险。

## 3. Recall 和 Precision 在 RAG 中的取舍

检索评估里常见两个目标：

- Recall：相关证据有没有被找回来。
- Precision：找回来的结果里有多少是真的相关。

RAG 的第一阶段检索更偏 recall：

```text
宁可候选池稍微多一点，也不要把正确证据漏掉。
```

但 context 阶段不能无限追求 recall。因为放进 prompt 的 chunk 太多会带来：

- token 成本增加。
- 噪声增加。
- 模型注意力被分散。
- 冲突证据增加。
- 引用更难对齐。

所以 RAG 检索通常是两段式目标：

```text
retrieval stage: high recall
context stage: high precision and evidence support
```

第三章关注 retrieval stage。第五章会继续讲 reranking 如何把候选池变成高质量上下文。

## 4. Sparse Retrieval：词项匹配路线

Sparse retrieval 把文本表示成稀疏词项向量。最经典代表是 BM25。

它依赖这样的直觉：

```text
如果 query 里的重要词在某个文档里出现，这个文档可能相关。
如果这个词在整个语料里很少见，它更有区分度。
如果文档特别长，要避免它因为词多而天然占优。
```

Sparse retrieval 特别适合：

- 错误码：`401`、`E_CONN_RESET`。
- 函数名：`refreshToken()`。
- 配置项：`max_retries`。
- 产品名、SKU、订单号。
- 法律条款号。
- 明确关键词查询。

它不擅长：

- 同义改写。
- 用户口语和文档术语不一致。
- 跨语言语义匹配。
- 需要概念泛化的问题。

## 5. 倒排索引

BM25 通常建立在倒排索引上。

正排索引像这样：

```text
doc_1 -> ["rag", "retriever", "generator"]
doc_2 -> ["bm25", "sparse", "retrieval"]
```

倒排索引反过来：

```text
"rag"       -> [(doc_1, freq=1), ...]
"retriever" -> [(doc_1, freq=1), ...]
"bm25"      -> [(doc_2, freq=1), ...]
```

当 query 是：

```text
"bm25 retrieval"
```

检索器只需要查 `bm25` 和 `retrieval` 的 postings list，而不需要扫描所有文档。

这也是 sparse retrieval 在工程上仍然重要的原因：它快、可解释、对精确词很强。

## 6. TF、IDF 和长度归一

BM25 可以看成 TF-IDF 思想的增强版本。

### TF：词频

如果 query 词在文档里出现更多次，文档可能更相关。

但词频不能线性增长。一个词出现 20 次不代表相关性是出现 2 次的 10 倍。BM25 使用饱和函数，让词频收益逐渐变小。

### IDF：逆文档频率

如果一个词在所有文档里都出现，它区分度很低。

例如：

- `the`、`and`、`system` 这类词通常区分度低。
- `GraphRAG`、`TokenExpired`、`HNSW` 区分度高。

IDF 的直觉：

```text
越稀有的词，命中时越重要。
```

### 文档长度归一

长文档因为包含更多词，更容易命中 query。BM25 会用文档长度做归一，避免长文档天然占优。

## 7. BM25 公式

常见 BM25 公式：

```text
score(q, d) = sum IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

其中：

- `t`：query 中的词项。
- `f(t,d)`：词项 `t` 在文档 `d` 中的出现次数。
- `IDF(t)`：词项 `t` 的稀有程度。
- `|d|`：文档长度。
- `avgdl`：平均文档长度。
- `k1`：控制词频饱和速度。
- `b`：控制文档长度归一强度。

### `k1` 怎么理解

`k1` 越大，词频增长对分数的影响越持久。

`k1` 越小，词频很快饱和。

常见范围大约在 `1.2 - 2.0`。很多系统直接用默认值，因为调参收益通常小于改 chunking、hybrid、rerank。

### `b` 怎么理解

`b = 0` 时，不做文档长度归一。

`b = 1` 时，完整使用长度归一。

常见默认值是 `0.75`。如果 chunk 长度差异很大，`b` 的影响会更明显；如果所有 chunk 长度接近，`b` 的影响较小。

### BM25 的工程启发

BM25 强依赖词项存在。因此：

- chunk 中保留标题、错误码、函数名很重要。
- query rewrite 可以把用户口语改成文档术语。
- 对中文、日文等语言，分词质量会显著影响 BM25。
- 对代码和配置文档，tokenizer 必须保留 `_`、`.`、`-` 等符号语义。

## 8. Sparse Retrieval 的现代变体

Sparse retrieval 不等于只有传统 BM25。

SPLADE 这类 learned sparse retrieval 会用神经模型生成稀疏词项表示。它保留 sparse 检索的可索引性，又能做一定程度的词项扩展。

直觉上：

```text
原始 query: "car"
扩展后 sparse terms: "car", "automobile", "vehicle"
```

它位于 BM25 和 dense retrieval 之间：

- 比 BM25 更能处理词汇不匹配。
- 比 dense retrieval 更保留词项可解释性。
- 但模型和索引复杂度更高。

本教程的 demo 不实现 SPLADE，但你应该知道 sparse retrieval 本身也在演进。

## 9. Dense Retrieval：向量语义路线

Dense retrieval 把 query 和 chunk 映射到向量空间：

```text
v_q = embed(q)
v_d = embed(d)
score = similarity(v_q, v_d)
```

如果两个文本语义接近，它们的向量距离应该更近。

例如：

```text
query: "How do I fix an expired token?"
chunk: "Refresh the access token and retry the request."
```

这两个文本可能没有大量词面重叠，但语义相关。BM25 可能错过，dense retrieval 更可能找回来。

DPR 这类 dense passage retrieval 工作证明了 dense retriever 可以在开放域问答中有效替代或补充传统 sparse retriever。后续 E5、BGE、GTE 等 embedding 模型进一步把 dense retrieval 变成 RAG 工程中的常用组件。

## 10. Embedding 空间怎么工作

embedding 模型把文本编码成固定维度向量：

```text
"BM25 is useful for exact term matching"
-> [0.12, -0.03, 0.88, ...]
```

向量维度本身通常不可直接解释，但模型训练目标会让语义相近文本在空间中更近。

常见训练方式包括：

- 对比学习：让相关 query-document 更近，不相关样本更远。
- 双塔结构：query encoder 和 document encoder 分别编码。
- 指令式 embedding：输入包含任务指令，例如 `Represent this sentence for retrieval`。
- 多语言对齐：让不同语言的同义文本向量接近。

工程上你不需要手写 embedding 模型，但必须理解它的约束：

- query 和 document 必须使用同一套兼容 embedding 模型。
- 换 embedding 模型通常要重建索引。
- embedding 维度必须和向量索引一致。
- 不同模型的相似度分数不可直接比较。

## 11. 相似度函数：cosine、dot product、L2

### Cosine similarity

余弦相似度看方向：

```text
cosine(a, b) = dot(a, b) / (||a|| * ||b||)
```

如果向量已归一化，cosine 和 dot product 排序通常等价。

### Dot product

点积同时受方向和向量长度影响：

```text
dot(a, b) = sum a_i * b_i
```

有些 embedding 模型训练时就是用 dot product 作为相似度。

### L2 distance

L2 距离衡量欧氏距离：

```text
L2(a, b) = sqrt(sum (a_i - b_i)^2)
```

对于归一化向量，cosine、dot product、L2 之间存在关系。但不要随意替换相似度函数。应该按照 embedding 模型说明和向量库配置使用。

## 12. Dense Retrieval 的优势和失败模式

优势：

- 能处理同义表达。
- 能处理语义相似。
- 对用户口语更友好。
- 可支持多语言检索。
- 对概念型问题更强。

失败模式：

- 漏掉精确标识符：`ERR-8492`、`refresh_token_v2`。
- 召回“语义像但不支持答案”的 chunk。
- 对领域术语理解不足。
- 对短 query 不稳定。
- embedding 模型和语料语言/领域不匹配。
- chunk 太长时语义被平均，细节信号被稀释。

最重要的一点：

```text
semantic similarity != answer support
```

语义相似只说明“看起来相关”。RAG 最终需要的是“能支持答案”。这也是为什么 dense retrieval 后经常还需要 reranking。

## 13. ANN 和向量库在本章中的位置

如果语料只有几千个 chunk，可以暴力计算 query 向量和所有 chunk 向量的相似度。

如果有几百万、几千万个 chunk，暴力搜索成本太高，就需要 ANN。

ANN 是 Approximate Nearest Neighbor，近似最近邻。它牺牲一点精确度，换取速度和规模。

常见方法：

- HNSW：图结构近邻搜索。
- IVF：先聚类，再搜索部分簇。
- PQ：压缩向量，降低内存。

第 2 章已经讲过 HNSW 的直觉。第 3 章只需要记住：

```text
dense retrieval 的语义能力来自 embedding 模型；
大规模 dense retrieval 的速度来自 ANN/vector index。
```

调 dense retrieval 时，不要把模型问题和索引问题混在一起：

- embedding 模型不适合：相关文档向量本来就不近。
- ANN 参数太激进：相关文档本来近，但近似搜索没找回来。
- metadata filter 配错：相关文档被过滤掉。

## 14. 为什么需要 Hybrid Retrieval

BM25 和 dense retrieval 各有盲区。

BM25 强在：

- 精确词。
- 稀有标识符。
- 错误码。
- 可解释性。
- 低成本 baseline。

Dense retrieval 强在：

- 同义表达。
- 语义泛化。
- 用户口语。
- 多语言。
- 概念型问题。

Hybrid retrieval 的目标是把两者结合：

```text
query
-> sparse retriever
-> dense retriever
-> merge candidates
-> deduplicate
-> fuse scores/ranks
-> final candidate list
```

生产 RAG 常用 hybrid，不是因为它时髦，而是因为用户问题分布混杂：

- 有人输入错误码。
- 有人输入自然语言描述。
- 有人输入产品名。
- 有人输入模糊症状。

单一路径很难覆盖所有查询形态。

## 15. Hybrid 的三种融合方式

### 15.1 Candidate union

最简单方式是取并集：

```text
candidates = bm25_top_k ∪ dense_top_k
```

优点：

- 简单。
- 召回高。
- 不需要分数归一。

缺点：

- 没有统一排序。
- 后面必须 rerank 或另行打分。

### 15.2 Score normalization + weighted fusion

把不同检索器的分数归一化，再加权：

```text
score(d) = alpha * norm_bm25(d) + beta * norm_dense(d)
```

问题是 BM25 分数和 cosine 分数不是同一种东西：

- BM25 分数范围随 query 和语料变化。
- cosine 分数通常在较小范围内。
- 不同 embedding 模型分数分布也不同。

所以 weighted fusion 需要校准。没有评估集时，权重很容易调成玄学。

### 15.3 Reciprocal Rank Fusion

RRF 只看排名，不直接比较原始分数。

公式：

```text
RRF(d) = sum 1 / (k + rank_i(d))
```

其中：

- `rank_i(d)` 是文档 `d` 在第 `i` 个检索器中的排名。
- `k` 是平滑常数，常见默认值约 60。

直觉：

- 如果一个文档在多个检索器里都靠前，它会得高分。
- 如果一个文档只在某个检索器里靠前，也能保留机会。
- 因为只用 rank，所以不要求 BM25 和 dense 分数可比。

RRF 很适合作为 hybrid retrieval 的默认融合方式。

## 16. RRF 示例

假设：

```text
BM25 ranking:
1. d03
2. d05
3. d01

Dense ranking:
1. d05
2. d04
3. d03
```

取 `k = 60`：

```text
RRF(d03) = 1/(60+1) + 1/(60+3)
RRF(d05) = 1/(60+2) + 1/(60+1)
RRF(d04) = 1/(60+2)
```

`d03` 和 `d05` 会因为同时出现在两个列表中而更稳。`d04` 只被 dense 检索到，也仍然保留。

这就是 hybrid 的核心价值：不同检索器相互补盲。

## 17. Metadata Filter 和权限过滤

检索不是只看文本相似度。真实 RAG 还要考虑：

- 用户权限。
- 租户。
- 语言。
- 文档类型。
- 产品线。
- 时间范围。
- 数据版本。

过滤有两种方式。

### Pre-filter

先过滤，再检索：

```text
allowed_docs = filter(D, user_acl, tenant, language)
retrieve(q, allowed_docs)
```

优点：

- 无权限文档不会进入候选。
- trace 更安全。
- 排名不会被无权限文档占坑。

缺点：

- 某些向量索引在强过滤下召回可能下降。
- 复杂 ACL join 会增加系统复杂度。

### Post-filter

先检索，再过滤：

```text
candidates = retrieve(q, D)
allowed = filter(candidates, user_acl)
```

风险：

- top-k 可能被无权限文档占满，过滤后剩余候选太少。
- reranker 或日志可能已经接触敏感内容。
- 检索分布会被污染。

生产系统默认应优先 pre-filter 或 query-time filter。post-filter 只适合作为补充保护，不应是唯一权限控制。

## 18. Top-k 和 Candidate Pool

常见错误是把 `top_k` 当成一个固定魔法数字。

不同阶段的 k 应该分开：

```text
sparse_top_k = 50
dense_top_k = 50
fused_top_k = 80
rerank_top_k = 20
context_top_k = 5
```

为什么 candidate pool 可以大，而 context 小？

因为第一阶段便宜，目标是召回；后面 rerank 和生成更贵，目标是精确。

调参建议：

- 先用评估集看 `Recall@k` 曲线。
- 如果 `Recall@10` 很低，问题在召回器或 chunking。
- 如果 `Recall@50` 高但最终答错，问题可能在 rerank/context/generation。
- 如果 context top-k 增大后答案变差，说明噪声污染了生成。

## 19. 检索失败模式

### 19.1 Top-k 没有正确证据

可能原因：

- 文档根本不存在。
- 文档未进入当前索引版本。
- chunking 切断答案。
- sparse 词项不匹配。
- dense embedding 不适合领域。
- metadata filter 过滤掉了正确文档。

优先排查：

```text
直接用关键词搜 source
-> 检查 chunk
-> 检查索引版本
-> 分别看 sparse 和 dense 结果
```

### 19.2 正确证据有，但排名太低

可能原因：

- top-k 太小。
- BM25 被高频词干扰。
- dense 召回很多语义近邻。
- fusion 权重或 RRF 排名不理想。
- 缺少 reranker。

解决方向：

- 扩大 candidate pool。
- 加 reranker。
- 调融合策略。
- 改 query rewrite。

### 19.3 BM25 找不到同义表达

例如：

```text
query: "登录凭证过期怎么办"
doc: "access token expired; refresh token and retry"
```

BM25 可能因为词面不匹配而失败。

解决方向：

- dense retrieval。
- query rewrite。
- 同义词词典。
- learned sparse retrieval。

### 19.4 Dense 漏掉精确标识符

例如：

```text
query: "ERR_AUTH_0042"
```

embedding 可能把这个看成普通字符串，不能稳定匹配。

解决方向：

- BM25。
- keyword boost。
- 字段索引。
- exact match fallback。

### 19.5 Hybrid 返回重复结果

可能原因：

- overlap 过大。
- 同一文档多段相似 chunk 同时命中。
- 多个检索器返回相同 parent。

解决方向：

- 按 `doc_id` 或 `parent_id` 去重。
- 保留最高分 child。
- context 阶段做 source 多样性控制。

## 20. 调参指南

### 20.1 BM25 参数

先用默认值：

```text
k1 = 1.2 ~ 2.0
b = 0.75
```

只有在有评估集时再调。没有评估集，调 BM25 参数通常不如改 chunking、tokenizer、hybrid。

### 20.2 Embedding 模型

选择 embedding 模型要看：

- 语言：中文、英文、多语言。
- 领域：通用、代码、法律、医学、金融。
- 上下文长度。
- 维度和存储成本。
- 是否支持 query/document 指令格式。
- 是否能本地部署。
- 是否需要重排器配套。

换 embedding 模型要重建向量索引。

### 20.3 相似度函数

不要随意换 cosine、dot product、L2。先看模型说明。

如果模型输出向量未归一化，cosine 和 dot product 可能给出不同排序。

### 20.4 RRF 常数

RRF 中的 `k` 控制排名贡献衰减。较大的 `k` 会让前几名之间差异变小，融合更平滑；较小的 `k` 更强调头部排名。

默认可以从 `60` 开始。调参仍然要看评估集。

### 20.5 Fusion 权重

如果业务中精确词非常重要，可以提高 sparse 权重。

如果用户问题很口语化，可以提高 dense 权重。

但加权融合必须用评估集验证，否则很容易过拟合少数样例。

## 21. Demo 讲解

运行：

```bash
python demos/03_sparse_dense_hybrid.py
```

代码：

```python
from rag_core import BM25Index, CORPUS, dense_style_search, print_ranking, reciprocal_rank_fusion

question = "How can semantic retrieval and keyword retrieval work together?"
bm25 = BM25Index(CORPUS).search(question, k=5)
dense = dense_style_search(question, k=5)
hybrid = reciprocal_rank_fusion([bm25, dense], limit=5)
```

对应关系：

| 代码 | 概念 |
| --- | --- |
| `BM25Index(CORPUS).search(...)` | sparse retrieval |
| `dense_style_search(...)` | 简化版 dense retrieval |
| `reciprocal_rank_fusion(...)` | hybrid retrieval / RRF |
| `print_ranking(...)` | 检索 trace |

### 为什么 dense-style 不是真实 embedding

本教程保持无依赖，所以 `dense_style_search` 没有调用真实 embedding 模型。它做的是：

```text
tokenize
-> term expansion
-> Counter vector
-> cosine similarity
```

这只是为了模拟“语义扩展后，query 和 chunk 即使词面不完全一致也能匹配”的效果。

真实 dense retrieval 会替换为：

```python
query_vector = embedding_model.embed_query(question)
doc_vectors = vector_store.search(query_vector, top_k=50)
```

但系统结构不变：

```text
query -> sparse
query -> dense
sparse + dense -> fusion
```

## 22. 从 toy demo 到真实系统

toy demo：

```text
BM25 in memory
dense-style Counter vector
RRF fusion
print result
```

真实系统：

```text
Elasticsearch / OpenSearch BM25
vector database / ANN index
metadata filter
RRF / weighted fusion
reranker
context builder
LLM generator
trace logger
```

替换路径：

1. 保留 `Document` schema。
2. 把 `BM25Index` 换成搜索引擎。
3. 把 `dense_style_search` 换成 embedding + vector DB。
4. 保留 RRF 融合逻辑。
5. 增加 reranker。
6. 保存每一步 trace。

文档 loading、结构解析和 chunking 的生产化示例见：

```bash
python demos/production_ingestion/run_pipeline.py
```

详细说明见 [生产化 ingestion demo](../demos/production_ingestion/README.md)。

## 23. 新语料库如何选择检索策略

如果语料包含大量：

- 错误码。
- 专有名词。
- API 名。
- 函数名。
- 配置项。

先做 BM25，并保证 tokenizer 正确。

如果用户问题经常：

- 很口语。
- 描述症状。
- 使用同义词。
- 跨语言。
- 问概念。

需要 dense retrieval。

如果你无法确定用户会怎么问，默认从 hybrid retrieval 开始：

```text
BM25 baseline + dense retrieval + RRF
```

这是生产 RAG 中很稳的起点。

## 24. 实践检查清单

检索 trace 至少记录：

- 原始用户问题。
- 实际检索 query。
- 检索器类型。
- 每个检索器 top-k。
- 每个 chunk 的 id、source、score、rank。
- fusion 后排名。
- metadata filter 条件。
- 最终进入 context 的 chunk。

评估至少看：

- `Recall@k`。
- `MRR`。
- `nDCG`。
- duplicate rate。
- no-answer rate。
- per-query latency。
- 每个检索器的独立贡献。

上线前问自己：

- 正确证据是否在 sparse top-k 中？
- 正确证据是否在 dense top-k 中？
- hybrid 是否真的提高 recall？
- top-k 增大是否带来噪声？
- 权限过滤是否在检索前或检索时发生？
- 检索分数是否可解释或可追踪？

## 25. 本章检查清单

学完本章，你应该能做到：

- 解释 BM25 的 TF、IDF、长度归一。
- 说明 `k1` 和 `b` 的直觉。
- 解释 dense retrieval 的 embedding、相似度和归一化。
- 区分 semantic similarity 和 answer support。
- 说明为什么 dense 会漏精确标识符。
- 说明为什么 BM25 会漏同义表达。
- 手写 RRF 公式，并解释为什么它不要求分数可比。
- 解释 pre-filter 和 post-filter 的差异。
- 运行 `demos/03_sparse_dense_hybrid.py` 并解释输出。
- 用 failure mode 表排查 RAG 检索问题。

## 参考资料

- Robertson and Zaragoza, 2009, The Probabilistic Relevance Framework: BM25 and Beyond: https://dblp.org/rec/journals/ftir/RobertsonZ09.html
- Karpukhin et al., 2020, Dense Passage Retrieval for Open-Domain Question Answering: https://arxiv.org/abs/2004.04906
- Khattab and Zaharia, 2020, ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT: https://arxiv.org/abs/2004.12832
- Thakur et al., 2021, BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models: https://arxiv.org/abs/2104.08663
- Formal et al., 2021, SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking: https://arxiv.org/abs/2107.05720
- Wang et al., 2022, Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5): https://arxiv.org/abs/2212.03533
- Cormack, Clarke, and Buettcher, 2009, Reciprocal Rank Fusion: https://doi.org/10.1145/1571941.1572114
- Gao et al., 2023, Retrieval-Augmented Generation for Large Language Models: A Survey: https://arxiv.org/abs/2312.10997
