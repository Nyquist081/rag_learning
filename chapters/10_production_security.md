# 第 10 章：生产化、安全与持续迭代

## 生产 RAG 关心什么

生产系统要回答的不只是“准不准”，还包括：

- 是否泄漏用户无权访问的数据。
- 是否能抵抗检索文本里的恶意指令。
- 是否能解释答案来源。
- 是否能回放一次请求。
- 是否能控制成本和延迟。
- 是否能随知识库更新稳定演进。

## 安全风险

截至 2026-05，RAG 安全重点包括：

- Prompt injection：检索到的文档里包含恶意指令。
- Data poisoning：攻击者把错误知识写入知识库。
- Sensitive information disclosure：模型输出敏感信息。
- Access-control bypass：向量库召回了用户无权限的 chunk。
- Vector and embedding weakness：embedding 空间被操纵或碰撞。
- Excessive agency：Agentic RAG 调用过多工具或越权操作。

OWASP LLM Top 10 2025 把 prompt injection、敏感信息泄漏、数据和模型投毒、向量与 embedding 弱点等列为核心风险。

## 防护策略

- 检索前做权限过滤。
- 对不可信来源打标签并降低优先级。
- prompt 明确区分“证据文本”和“系统指令”。
- 不执行检索内容中的指令。
- 对关键输出做 claim verification。
- 对索引增量更新做来源审计。
- 对敏感字段做脱敏。
- 保存完整 trace。

## Demo

运行：

```bash
python demos/10_security_filtering.py
```

demo 加入一个含恶意指令的 `untrusted` 文档，然后展示不过滤和过滤后的检索结果差异。

## 监控指标

- 请求量、延迟、token 成本。
- 检索 top-k 分数分布。
- fallback rate。
- 拒答率。
- 用户纠错率。
- 引用缺失率。
- 权限过滤命中率。
- 安全拦截率。

## 参考

- OWASP Top 10 for LLM Applications: https://owasp.org/www-project-top-10-for-large-language-model-applications
- PoisonedRAG, 2024: https://arxiv.org/abs/2402.07867
- BadRAG, 2024: https://arxiv.org/abs/2406.00083
- SD-RAG, 2026: https://arxiv.org/abs/2601.11199

