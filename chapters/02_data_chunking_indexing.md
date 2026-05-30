# 第 2 章：数据、切分与索引

本章目标是把 RAG 的数据层和索引层讲透。读完后，你应该能回答：

- 为什么 RAG 质量首先取决于数据，而不是模型？
- 一个可上线的 chunk 应该保存哪些字段？
- 固定长度切分、递归切分、语义切分、parent-child、late chunking 的原理分别是什么？
- chunk size 和 overlap 为什么会影响召回和生成？
- BM25 索引、向量索引、HNSW/ANN 索引分别解决什么问题？
- 为什么要记录索引版本、chunker 版本、embedding 版本？

## 1. 为什么数据层是 RAG 的地基

RAG 的在线回答链路看起来是：

```text
question -> retrieve -> context -> generate answer
```

但真正决定质量的是更早的离线链路：

```text
raw data
-> parse
-> clean
-> normalize
-> chunk
-> attach metadata
-> embed / index
-> publish index version
```

如果离线链路做得不好，在线链路会出现这些问题：

- 文档里明明有答案，但 chunk 被切断，检索不到完整证据。
- 召回了正文，却丢了标题、章节、表头，模型不知道上下文。
- 检索到了旧版本文档，答案过期。
- 检索到了用户无权限访问的 chunk，造成数据泄漏。
- 同一个文档重复进入索引，top-k 被重复结果污染。
- embedding 模型更新后混用了旧向量和新向量，距离分数失真。

因此，RAG 工程里一个重要原则是：

```text
先保证数据可用、可追溯、可版本化，再优化检索和生成。
```

## 2. 从原始文档到可检索对象

原始文档通常不是天然适合检索的。它可能来自：

- Markdown / HTML / PDF。
- Word / Excel / PPT。
- 数据库记录。
- 工单、聊天记录、邮件。
- 代码仓库。
- 日志和监控事件。
- API 文档和 OpenAPI schema。

这些数据需要先变成统一的内部对象。

### 2.1 原始文档对象

一个原始文档可以表示为：

```python
RawDocument(
    doc_id="policy-2026-001",
    source_uri="s3://kb/hr/policy.md",
    content_type="text/markdown",
    raw_bytes=b"...",
    updated_at="2026-05-01T10:00:00Z",
    acl=["hr", "manager"],
)
```

这里的重点不是字段名，而是保留四类信息：

- 内容：文档正文。
- 来源：它从哪里来。
- 时间：它是什么版本。
- 权限：谁可以看。

### 2.2 解析后的结构化文档

解析后最好保留结构：

```python
ParsedDocument(
    doc_id="policy-2026-001",
    title="Employee Travel Policy",
    sections=[
        Section(path=["Employee Travel Policy", "Reimbursement"], text="..."),
        Section(path=["Employee Travel Policy", "International Travel"], text="..."),
    ],
    tables=[...],
    code_blocks=[...],
    metadata={...},
)
```

为什么要保留结构？

- 标题路径能帮助 chunk 在脱离全文后仍然有语义。
- 表格列名是事实的一部分，不能丢。
- 代码块语言、函数名、类名会影响检索。
- 页码、段落号、条款号用于引用和审计。

很多 RAG 系统失败，是因为把所有文档先“拍平成纯文本”，再粗暴切分，导致结构信息消失。

## 3. 清洗不是删除，而是保留有用信号

清洗的目标不是让文本“看起来干净”，而是让检索和生成能稳定使用。

### 3.1 应该删除的内容

- 页面导航。
- 页脚版权。
- 重复目录。
- 广告。
- 空白噪声。
- OCR 乱码。
- 跟正文无关的脚本和样式。

### 3.2 不应该轻易删除的内容

- 标题。
- 小标题。
- 列表编号。
- 表头。
- 图片 caption。
- 代码块。
- 错误码。
- 配置项。
- 日期和版本号。
- 法律条款号。

这些信息对检索很重要。比如用户问“401 Token Expired 怎么处理”，`401`、`Token Expired`、错误码表头都可能是最强检索信号。

### 3.3 清洗算法的基本流程

```text
parse raw document
-> remove boilerplate
-> normalize whitespace
-> normalize unicode
-> preserve structural markers
-> detect language / type
-> output structured text blocks
```

伪代码：

```python
def clean(blocks):
    cleaned = []
    for block in blocks:
        if is_boilerplate(block):
            continue
        text = normalize_unicode(block.text)
        text = normalize_whitespace(text)
        if is_empty(text):
            continue
        cleaned.append(block.with_text(text))
    return cleaned
```

注意：清洗规则必须可复现。否则索引重建后，同一个文档可能生成不同 chunk，评估无法对比。

## 4. Chunk 是 RAG 的检索单位

chunk 是检索系统能返回的最小证据单元。

如果 chunk 太小：

- 召回更精确。
- 但缺少上下文。
- 生成时可能答不完整。
- 相关信息可能散落在多个 chunk。

如果 chunk 太大：

- 上下文更完整。
- 但检索分数更模糊。
- top-k 更容易被长文档占据。
- token 成本更高。
- 重排更慢。

所以 chunking 本质是在优化一个 trade-off：

```text
retrieval precision vs context completeness
```

RAG 里没有一个永远正确的 chunk size。合理的 chunking 必须和文档类型、问题类型、检索器、上下文窗口、评估集一起调。

## 5. 固定长度切分

固定长度切分是最简单的方法：

```text
每 N 个 token 切一段，相邻段保留 M 个 token overlap
```

伪代码：

```python
def fixed_window(tokens, size=400, overlap=80):
    step = size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        chunks.append(tokens[start : start + size])
    return chunks
```

### 5.1 为什么需要 overlap

因为答案可能跨越切分边界。

例如：

```text
chunk A: Token Expired usually happens when the access token lifetime is exceeded.
chunk B: The fix is to refresh the token and retry the request.
```

如果没有 overlap，用户问“Token Expired 怎么修复”时，A 有原因但没有修复，B 有修复但缺主语。overlap 能缓解这个问题。

### 5.2 固定切分的优点

- 实现简单。
- 分布稳定。
- 容易批量处理。
- 适合没有明显结构的文本。

### 5.3 固定切分的缺点

- 可能切断句子、表格、代码块。
- 不理解标题层级。
- 对文档结构强的材料效果差。
- 同一个 chunk 可能混入多个话题。

固定长度切分适合作为 baseline，不适合直接当成所有数据的最终方案。

## 6. 递归结构切分

递归切分会按照结构优先级切文档：

```text
标题
-> 段落
-> 句子
-> token
```

算法思想：

1. 先尝试按最大语义边界切分，比如一级/二级标题。
2. 如果某段仍然太长，再按段落切。
3. 如果段落仍然太长，再按句子切。
4. 最后才按 token 硬切。

伪代码：

```python
def recursive_split(text, max_tokens):
    for separator in ["\n## ", "\n\n", "。", ". ", " "]:
        parts = split_by(text, separator)
        if all(token_count(part) <= max_tokens for part in parts):
            return parts
    return fixed_window(tokenize(text), size=max_tokens)
```

### 为什么递归切分通常更好

因为它优先尊重人类写作结构。标题、段落和句子往往就是语义边界。

技术文档、产品文档、法律条款、操作手册通常都适合递归结构切分。

## 7. 语义切分

语义切分不是按长度或标点，而是按“话题是否发生变化”切。

常见算法：

1. 把文档切成句子或小段。
2. 给每个句子/小段计算 embedding。
3. 计算相邻片段相似度。
4. 当相似度明显下降时，认为出现话题边界。
5. 合并边界之间的片段形成 chunk。

公式上，可以计算相邻向量余弦相似度：

```text
sim_i = cosine(embed(segment_i), embed(segment_{i+1}))
```

如果：

```text
sim_i < threshold
```

则在 `i` 和 `i+1` 之间切分。

### 语义切分的优点

- 能减少一个 chunk 混入多个话题。
- 对长篇叙述、会议纪要、邮件线程更自然。
- 有时能提高 dense retrieval 的召回质量。

### 语义切分的成本和风险

- 需要提前计算 embedding。
- 阈值难调。
- 不同 embedding 模型给出的边界可能不同。
- 语义边界不一定等于问答需要的证据边界。
- 有研究指出，语义切分并不总是稳定优于简单 fixed-size chunking，需要用评估集验证。

因此语义切分不应该被当作“高级所以一定更好”。它只是候选策略之一。

## 8. Parent-child chunking

Parent-child 是生产 RAG 里很常用的策略。

核心思想：

```text
小 child chunk 用于检索
大 parent chunk 用于生成上下文
```

为什么这样做？

- 小 chunk 检索更精确。
- 大 chunk 保留更完整上下文。
- 两者结合能缓解“召回精确 vs 上下文完整”的冲突。

结构示例：

```text
parent section: 退款政策完整章节，1200 tokens
  child chunk 1: 适用范围，300 tokens
  child chunk 2: 退款条件，300 tokens
  child chunk 3: 审批流程，300 tokens
```

检索流程：

```text
query
-> retrieve child chunks
-> map child.parent_id
-> fetch parent sections
-> dedupe parents
-> send parent or stitched context to generator
```

伪代码：

```python
child_hits = search_child_index(query)
parent_ids = unique(hit.parent_id for hit in child_hits)
contexts = load_parent_chunks(parent_ids)
answer = generate(query, contexts)
```

适用场景：

- 技术文档。
- 法律条款。
- 产品手册。
- API 文档。
- 长章节问答。

风险：

- parent 太大时会重新引入噪声。
- 多个 child 命中同一个 parent 时要去重。
- 引用最好能指向 child，同时上下文来自 parent。

## 9. Small-to-big 与邻居扩展

Small-to-big 是 parent-child 的近亲：

```text
先召回小 chunk，再向前后扩展邻居 chunk
```

例如召回 `chunk_10` 后，把 `chunk_9`、`chunk_10`、`chunk_11` 一起放入上下文。

适合：

- 答案经常跨相邻段落。
- 文档顺序很重要。
- 不想维护 parent-child 双索引。

注意：

- 邻居扩展必须受 token budget 控制。
- 扩展前后要保留 source 和顺序。
- 不能把无权限邻居带进上下文。

## 10. Late Chunking

传统 chunking 是：

```text
先切 chunk -> 再分别 embed 每个 chunk
```

Late chunking 的思想是：

```text
先用长上下文 embedding 模型处理完整文档
-> 得到上下文化 token 表示
-> 再按 chunk 边界 pooling 得到 chunk embedding
```

这样做的动机是：传统方法里，每个 chunk 在 embedding 时只看到自己，可能丢失跨 chunk 的上下文。Late chunking 让 chunk embedding 在生成前已经吸收了更长文档上下文。

适合：

- embedding 模型支持较长上下文。
- chunk 本身需要前文才能理解。
- 文档内部有强上下文依赖。

限制：

- 实现复杂度更高。
- 对 embedding 模型能力和上下文长度有要求。
- 计算成本可能更高。
- 并不替代 source、metadata 和权限过滤。

截至 2026-05，late chunking 是值得关注的新方向，但工程上仍要和传统 chunking 在自己的评估集上对比。

## 11. RAPTOR 与层级索引

RAPTOR 不是普通切分策略，而是层级检索结构。

流程：

```text
leaf chunks
-> embed chunks
-> cluster chunks
-> summarize each cluster
-> embed summaries
-> repeat recursively
-> build summary tree
```

检索时可以同时检索：

- 叶子节点：细节证据。
- 中间摘要：局部主题。
- 根部摘要：全局视角。

为什么有用？

普通 RAG 通常只检索短的连续 chunk。对于需要全局理解、多跳推理、跨章节总结的问题，短 chunk 可能太局部。RAPTOR 用摘要树补充了层级语义。

代价：

- 需要 LLM 生成摘要。
- 索引构建成本更高。
- 摘要可能引入错误。
- 引用最终仍要能回到叶子 chunk。

第 7 章会进一步讲层级检索和 multi-hop。

## 12. 表格、代码和多模态文档怎么切

### 表格

不要直接把表格拍平成一长串文本。至少保留：

- 表名。
- 列名。
- 行主键。
- 单元格值。
- 单位。
- 页码或 section。

一个表格行 chunk 可以这样表达：

```text
Table: API Error Codes
Columns: code, message, fix
Row: code=401, message=Token Expired, fix=Refresh token and retry
```

### 代码

代码适合按结构切：

- 文件。
- class。
- function。
- docstring。
- signature。
- imports。

代码 chunk 必须保留文件路径和行号，否则用户无法定位。

### PDF

PDF 的主要难点：

- 页面顺序可能错乱。
- 页眉页脚重复。
- 多栏布局。
- 表格抽取不稳定。
- OCR 错字。

PDF RAG 的重点往往不是 embedding，而是解析质量。

## 13. 元数据设计

一个生产 chunk 至少应该包含：

```python
Chunk(
    chunk_id="policy-2026-001#section-3#chunk-2",
    parent_id="policy-2026-001#section-3",
    doc_id="policy-2026-001",
    source_uri="s3://kb/hr/policy.md",
    title_path=["Employee Travel Policy", "Reimbursement"],
    text="...",
    chunk_index=2,
    token_count=384,
    updated_at="2026-05-01T10:00:00Z",
    acl=["hr", "manager"],
    content_hash="...",
    parser_version="pdf-parser-v3",
    chunker_version="recursive-v2-size400-overlap80",
    embedding_model="text-embedding-model-x",
    embedding_dim=1536,
)
```

字段作用：

| 字段 | 作用 |
| --- | --- |
| `chunk_id` | 唯一定位 chunk |
| `parent_id` | parent-child 或邻居扩展 |
| `doc_id` | 文档级去重和版本管理 |
| `source_uri` | 引用和审计 |
| `title_path` | 保留结构上下文 |
| `chunk_index` | 恢复原始顺序 |
| `acl` | 权限过滤 |
| `content_hash` | 增量更新和去重 |
| `parser_version` | 排查解析变化 |
| `chunker_version` | 排查切分变化 |
| `embedding_model` | 防止向量混用 |

## 14. 索引类型

RAG 常见索引不是只有向量索引。

### 14.1 倒排索引

BM25 通常基于倒排索引：

```text
term -> [(doc_id, term_frequency), ...]
```

用户 query 进入后：

1. 分词。
2. 找到每个 term 对应的 postings list。
3. 计算每个候选文档的 BM25 分数。
4. 返回 top-k。

BM25 分数直觉：

```text
score = IDF(term) * TF_saturation(term, doc) * length_normalization(doc)
```

它适合精确词、编号、错误码、产品名。

### 14.2 向量索引

向量索引存储：

```text
chunk_id -> embedding vector
```

query 也会被 embed 成向量，然后做最近邻搜索：

```text
score = cosine(embed(query), embed(chunk))
```

向量检索适合语义相似和同义表达。

### 14.3 ANN 索引

如果向量很多，暴力计算所有向量距离太慢。Approximate Nearest Neighbor 会牺牲一点精度换速度。

常见结构：

- HNSW：基于多层小世界图。
- IVF：先聚类，再只搜索部分簇。
- PQ：向量压缩，减少内存。

HNSW 的直觉：

```text
高层图用于快速跳转到大致区域
低层图用于局部精细搜索
```

查询时从高层入口开始贪心移动，逐层下降，最后在底层找近邻。

核心参数：

- `M`：每个节点保留多少邻居，影响内存和召回。
- `efConstruction`：建图搜索宽度，影响构建时间和质量。
- `efSearch`：查询搜索宽度，影响延迟和召回。

### 14.4 混合索引

实际 RAG 常同时维护：

- BM25 索引。
- 向量索引。
- 元数据过滤索引。
- parent-child 映射。
- source/version 表。

后续第 3 章会讲 hybrid retrieval 如何融合 sparse 和 dense 结果。

## 15. 增量更新与版本管理

生产知识库会持续变化。你需要回答：

- 哪些文档新增了？
- 哪些文档删除了？
- 哪些文档更新了？
- 哪些 chunk 需要重新 embed？
- 当前线上使用哪个索引版本？
- 某个历史答案当时用了哪个索引版本？

推荐流程：

```text
fetch source snapshot
-> compute content hash
-> detect added/updated/deleted docs
-> parse changed docs
-> chunk changed docs
-> embed changed chunks
-> build candidate index
-> run regression eval
-> publish index version
-> keep rollback version
```

索引版本可以命名为：

```text
kb=hr_policy
snapshot=2026-05-29T08:00:00Z
parser=markdown-v2
chunker=recursive-400-80-v3
embedding=bge-m3-v1
index=hnsw-M32-efC200
```

没有版本管理，RAG 的线上问题很难复现。

## 16. 权限过滤应该发生在哪里

原则：

```text
用户无权看到的 chunk，不应该进入候选集。
```

最好在检索前或检索时过滤：

```text
query + user_acl
-> permission-filtered retrieval
-> rerank allowed candidates
-> generate answer
```

不要先召回全量，再把无权限结果简单删除。原因：

- top-k 位置可能被无权限文档占掉，导致有权限文档没进候选。
- trace 中可能已经暴露了敏感 source。
- reranker 或 LLM judge 可能看到了敏感内容。

向量数据库、搜索引擎、业务数据库都要能支持 metadata filter 或 ACL join。

## 17. 如何选择 chunking 策略

不要凭感觉选。用评估集选。

### 17.1 从 baseline 开始

先建立两个 baseline：

- 固定长度：如 400 tokens，overlap 80。
- 结构递归：标题/段落/句子/token。

### 17.2 用评估指标比较

每种 chunking 策略至少看：

- `Recall@k`：正确证据是否能进 top-k。
- `MRR`：正确证据排得是否靠前。
- context token cost：上下文成本。
- duplicate rate：重复 chunk 比例。
- citation quality：引用是否精准。

### 17.3 按文档类型选择

| 文档类型 | 推荐起点 |
| --- | --- |
| FAQ | 小 chunk，按问答对切 |
| Markdown 技术文档 | 标题递归切分 |
| 法律条款 | 条款号/章节层级切分 |
| 表格 | 行/实体粒度 + 表头 |
| 代码 | function/class/file 层级 |
| 长报告 | parent-child 或 RAPTOR |
| 聊天记录 | 按会话/话题窗口 |

## 18. Demo 讲解

运行：

```bash
python demos/02_chunking_metadata.py
```

输出示例：

```text
id=d01 source=notes/rag_basics.md tags=basics title=RAG definition
```

这个 demo 很小，但它强调一个核心点：chunk 不是一段裸文本。每个 chunk 都要能回答：

- 我是谁？`id`
- 我来自哪里？`source`
- 我属于什么类型？`tags`
- 我在原文中的标题是什么？`title`

下一步练习：

1. 在 `demos/rag_core.py` 里新增 `parent_id`、`updated_at`、`chunk_index` 字段。
2. 把 `CORPUS` 替换成你自己的 Markdown 文件。
3. 写一个结构递归 chunker。
4. 对比固定切分和递归切分的 `Recall@k`。

### 18.1 生产化 ingestion demo

本仓库已经补充了一条更接近生产环境的数据管线：

```bash
python demos/production_ingestion/run_pipeline.py
```

代码入口：

- [run_pipeline.py](../demos/production_ingestion/run_pipeline.py)
- [ingestion_pipeline.py](../demos/production_ingestion/ingestion_pipeline.py)
- [README.md](../demos/production_ingestion/README.md)

它展示了：

- 目录扫描和 loader 路由。
- Markdown、HTML、JSON、TXT 加载。
- Markdown 标题层级、代码块、表格保留。
- 结构优先递归切分。
- token overlap。
- `doc_id`、`chunk_id`、source、hash、版本等 metadata。
- manifest 和增量更新判断。
- JSONL chunk 快照输出。

这套 demo 刻意只依赖 Python 标准库。真实生产系统里，可以把 loader 换成 Unstructured、Docling、LlamaParse，把 JSONL sink 换成 Elasticsearch、OpenSearch 或向量数据库。

## 19. 常见问题排查

### 问题 1：文档有答案，但检索不到

排查：

- chunk 是否太大或太小？
- 标题是否丢失？
- query 里的关键词是否出现在 chunk 中？
- 是否需要 dense retrieval 或 query rewrite？
- 文档是否进入了当前索引版本？

### 问题 2：检索结果重复

排查：

- overlap 是否太大？
- 是否有重复文档？
- parent-child 映射是否去重？
- 近重复 chunk 是否需要 hash 或 simhash 去重？

### 问题 3：答案引用不精确

排查：

- chunk 是否过大？
- 是否引用 parent 而不是 child？
- 是否保留了页码、标题、段落号？

### 问题 4：更新文档后答案仍然旧

排查：

- source snapshot 是否更新？
- content hash 是否变化？
- 旧 chunk 是否删除？
- embedding 是否重算？
- 线上是否切到新索引版本？

## 20. 本章检查清单

学完本章，你应该能做到：

- 解释为什么 chunk 是 RAG 的最小证据单元。
- 写出固定长度切分和递归切分的伪代码。
- 说明 overlap 的作用和副作用。
- 区分语义切分、parent-child、late chunking、RAPTOR。
- 解释 BM25 倒排索引和向量索引的基本结构。
- 说明 HNSW 为什么能加速向量检索。
- 设计一个带 source、ACL、version 的 chunk schema。
- 知道如何用评估集选择 chunking 策略。

## 参考资料

- Robertson and Zaragoza, 2009, The Probabilistic Relevance Framework: BM25 and Beyond: https://dblp.org/rec/journals/ftir/RobertsonZ09.html
- Malkov and Yashunin, 2016, HNSW: https://arxiv.org/abs/1603.09320
- Johnson et al., 2017, Billion-scale similarity search with GPUs / FAISS: https://arxiv.org/abs/1702.08734
- RAPTOR, 2024: https://arxiv.org/abs/2401.18059
- Late Chunking, 2024: https://arxiv.org/abs/2409.04701
- Is Semantic Chunking Worth the Computational Cost?, 2024: https://arxiv.org/abs/2410.13070
