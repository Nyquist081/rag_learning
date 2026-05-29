# 第 8 章：GraphRAG、LightRAG 与图结构检索

## 为什么需要图

普通 RAG 把知识库看成 chunk 列表。GraphRAG 把文本进一步抽取成实体、关系、社区和摘要，适合回答关系密集或全局洞察问题。

适合：

- 组织关系。
- 系统依赖。
- 事故根因网络。
- 多事件共性分析。
- 风险主题聚类。

不适合：

- 小规模 FAQ。
- 只需要精确事实查找。
- 实体体系不稳定。

## GraphRAG 流程

```text
documents
-> entity extraction
-> relation extraction
-> graph construction
-> community detection
-> community summary
-> local/global search
```

Microsoft GraphRAG 在 2024 年系统化推动了这条路线，后续还有 LazyGraphRAG、DRIFT Search 等降低成本和提升搜索质量的工作。LightRAG 则强调更轻量的图结构索引与检索。

## Demo

运行：

```bash
python demos/08_graph_rag.py
```

demo 用一个小图从 `GraphRAG` 节点扩展到 `entities`、`relationships`、`communities`，再把图邻域加入检索 query。

## 工程要点

- 实体抽取要有同义归一。
- 关系要保留来源证据。
- 社区摘要要能回溯到底层 chunk。
- 图检索和文本检索通常要结合。
- 图构建成本高，适合离线或增量更新。

## 参考

- Microsoft GraphRAG: https://www.microsoft.com/en-us/research/project/graphrag/
- LightRAG, 2024: https://arxiv.org/abs/2410.05779

