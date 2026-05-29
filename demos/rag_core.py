"""章节 demo 共用的玩具级 RAG 组件。

这份代码刻意不引入第三方依赖，用来演示 RAG 的核心机制：
切分、稀疏检索、类 dense 检索、融合、重排、纠错、多跳规划、
图遍历、证据约束生成和指标评估。
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class Document:
    """知识库中的一个最小文档片段。

    真实 RAG 系统里，这通常对应一个 chunk。除了正文以外，必须保留
    `id` 和 `source`，否则后续无法做引用、评估和日志回放。
    """

    id: str
    title: str
    text: str
    source: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredDocument:
    """带分数的检索结果。

    `reason` 用来记录这个结果来自哪个检索/排序阶段，方便观察 trace。
    """

    doc: Document
    score: float
    reason: str


CORPUS = [
    Document(
        "d01",
        "RAG definition",
        "RAG combines a retriever with a generator. The retriever fetches evidence from an external knowledge base, and the generator answers using that evidence.",
        "notes/rag_basics.md",
        ("basics",),
    ),
    Document(
        "d02",
        "Chunking strategy",
        "Chunking controls the retrieval unit. Good chunks preserve title paths, section boundaries, source ids, and enough local context for answer generation.",
        "notes/chunking.md",
        ("indexing",),
    ),
    Document(
        "d03",
        "BM25 retrieval",
        "BM25 is a sparse retrieval method. It works well for exact terms, error codes, product names, identifiers, and rare keywords.",
        "notes/sparse.md",
        ("retrieval", "sparse"),
    ),
    Document(
        "d04",
        "Dense embedding retrieval",
        "Dense retrieval embeds questions and chunks into vectors. It is strong for semantic similarity and paraphrases, but can miss exact identifiers.",
        "notes/dense.md",
        ("retrieval", "dense"),
    ),
    Document(
        "d05",
        "Hybrid retrieval",
        "Hybrid retrieval combines sparse retrieval and dense retrieval. Reciprocal rank fusion is a common way to merge ranked candidate lists.",
        "notes/hybrid.md",
        ("retrieval", "hybrid"),
    ),
    Document(
        "d06",
        "Reranking",
        "Reranking scores candidate chunks again with a stronger model. It should prefer chunks that directly support the answer, not just chunks on the same topic.",
        "notes/rerank.md",
        ("ranking",),
    ),
    Document(
        "d07",
        "HyDE query expansion",
        "HyDE generates a hypothetical answer or document, embeds that text, and retrieves real documents similar to it. It can help zero-shot dense retrieval.",
        "notes/hyde.md",
        ("query",),
    ),
    Document(
        "d08",
        "Corrective RAG",
        "Corrective RAG grades retrieved evidence. If the evidence is weak or irrelevant, it rewrites the query, searches again, or refuses to answer.",
        "notes/crag.md",
        ("advanced", "correction"),
    ),
    Document(
        "d09",
        "Self-RAG",
        "Self-RAG makes the model decide when to retrieve and critique whether retrieved passages are relevant, supported, and useful.",
        "notes/self_rag.md",
        ("advanced", "reflection"),
    ),
    Document(
        "d10",
        "GraphRAG",
        "GraphRAG extracts entities and relationships from text. It supports global questions by searching graph communities and local neighborhoods.",
        "notes/graphrag.md",
        ("advanced", "graph"),
    ),
    Document(
        "d11",
        "RAPTOR",
        "RAPTOR builds a tree of clustered chunks and summaries. It helps retrieval over long documents where both details and high-level summaries matter.",
        "notes/raptor.md",
        ("advanced", "hierarchy"),
    ),
    Document(
        "d12",
        "RAG evaluation",
        "RAG systems should evaluate retrieval and generation separately. Useful metrics include recall at k, MRR, nDCG, faithfulness, answer relevance, and citation support.",
        "notes/evaluation.md",
        ("evaluation",),
    ),
    Document(
        "d13",
        "RAG security",
        "Production RAG must handle prompt injection, poisoned documents, sensitive data leakage, access control, and vector or embedding weaknesses.",
        "notes/security.md",
        ("production", "security"),
    ),
]


RELATED_TERMS = {
    "semantic": {"dense", "embedding", "paraphrases", "similarity"},
    "keyword": {"bm25", "sparse", "exact", "terms"},
    "weak": {"irrelevant", "insufficient", "refuses", "rewrite"},
    "graph": {"entities", "relationships", "communities", "neighborhoods"},
    "evaluate": {"metrics", "recall", "faithfulness", "mrr", "ndcg"},
    "security": {"injection", "poisoned", "leakage", "access", "control"},
}


GRAPH_EDGES = {
    "RAG": {"retriever", "generator", "evidence"},
    "retriever": {"BM25", "dense retrieval", "hybrid retrieval", "reranking"},
    "hybrid retrieval": {"BM25", "dense retrieval", "RRF"},
    "advanced RAG": {"Corrective RAG", "Self-RAG", "GraphRAG", "RAPTOR"},
    "GraphRAG": {"entities", "relationships", "communities"},
    "production RAG": {"evaluation", "security", "observability"},
}


def tokenize(text: str) -> list[str]:
    """把输入文本切成小写 token。

    这里支持英文词、数字、下划线和单个中文字符。真实系统通常会换成
    更专业的 tokenizer 或 embedding 模型自带的分词逻辑。
    """

    return [token.lower() for token in TOKEN_RE.findall(text)]


def expanded_terms(text: str) -> list[str]:
    """给文本 token 做极简同义扩展。

    这不是生产级语义检索，只是为了在无依赖 demo 中模拟 embedding
    能捕捉语义相近词的效果。
    """

    terms = tokenize(text)
    expanded = list(terms)
    for term in terms:
        expanded.extend(RELATED_TERMS.get(term, set()))
    return expanded


def lexical_overlap(query: str, doc: Document) -> int:
    """计算 query 和文档之间的词项重叠数量。

    这个小函数会被重排和证据分级复用，用来模拟“直接相关性”的弱信号。
    """

    q = set(tokenize(query))
    d = set(tokenize(doc.title + " " + doc.text))
    return len(q & d)


class BM25Index:
    """一个极简 BM25 稀疏检索索引。

    BM25 适合做 RAG baseline：它不依赖模型，对关键词、错误码、专有名词
    很有效，而且分数来源相对可解释。
    """

    def __init__(self, docs: list[Document]) -> None:
        """预计算每个文档的词频、文档长度和全局文档频率。"""

        self.docs = docs
        self.term_freqs = [Counter(tokenize(doc.title + " " + doc.text)) for doc in docs]
        self.doc_lengths = [sum(freq.values()) for freq in self.term_freqs]
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_freqs = Counter()
        for term_freq in self.term_freqs:
            self.doc_freqs.update(term_freq.keys())

    def search(self, query: str, k: int = 5) -> list[ScoredDocument]:
        """用 BM25 分数检索 top-k 文档。"""

        terms = tokenize(query)
        scored = []
        for idx, doc in enumerate(self.docs):
            score = self._score(terms, idx)
            if score > 0:
                scored.append(ScoredDocument(doc, score, "bm25"))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def _score(self, terms: Iterable[str], idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """计算单个文档对 query terms 的 BM25 分数。

        分数由三部分组成：词频、逆文档频率、文档长度归一化。
        """

        score = 0.0
        tf = self.term_freqs[idx]
        length = self.doc_lengths[idx]
        total = len(self.docs)
        for term in terms:
            if term not in tf:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            numerator = tf[term] * (k1 + 1)
            denominator = tf[term] + k1 * (1 - b + b * length / self.avg_length)
            score += idf * numerator / denominator
        return score


def dense_style_search(query: str, docs: list[Document] = CORPUS, k: int = 5) -> list[ScoredDocument]:
    """用词项扩展和余弦相似度模拟一个极简 embedding 检索器。"""

    q = Counter(expanded_terms(query))
    scored = []
    for doc in docs:
        d = Counter(expanded_terms(doc.title + " " + doc.text))
        score = cosine(q, d)
        if score > 0:
            scored.append(ScoredDocument(doc, score, "dense_style"))
    return sorted(scored, key=lambda item: item.score, reverse=True)[:k]


def cosine(left: Counter[str], right: Counter[str]) -> float:
    """计算两个稀疏向量的余弦相似度。

    在真实 dense retrieval 中，向量通常来自 embedding 模型；这里用
    Counter 模拟向量，方便展示相似度计算的本质。
    """

    common = set(left) & set(right)
    numerator = sum(left[t] * right[t] for t in common)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def reciprocal_rank_fusion(rankings: list[list[ScoredDocument]], k: int = 60, limit: int = 5) -> list[ScoredDocument]:
    """用 RRF 融合多个检索器的排名结果。

    RRF 不要求 BM25 分数和 dense 分数处在同一尺度，只看每个结果在各自
    排名中的位置，因此很适合作为 hybrid retrieval 的简单融合方法。
    """

    scores: dict[str, float] = defaultdict(float)
    docs: dict[str, Document] = {}
    reasons: dict[str, list[str]] = defaultdict(list)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.doc.id] += 1 / (k + rank)
            docs[item.doc.id] = item.doc
            reasons[item.doc.id].append(item.reason)
    fused = [
        ScoredDocument(docs[doc_id], score, "rrf:" + "+".join(sorted(set(reasons[doc_id]))))
        for doc_id, score in scores.items()
    ]
    return sorted(fused, key=lambda item: item.score, reverse=True)[:limit]


def rerank_for_support(query: str, candidates: list[ScoredDocument], k: int = 5) -> list[ScoredDocument]:
    """用规则模拟“证据支持度”重排。

    真实系统通常用 cross-encoder、LLM judge 或学习排序模型。这里用标题
    重叠、正文重叠和少量支持性动词模拟，让 demo 保持无依赖。
    """

    reranked = []
    for item in candidates:
        title_bonus = lexical_overlap(query, Document(item.doc.id, item.doc.title, "", item.doc.source))
        text_overlap = lexical_overlap(query, item.doc)
        direct_support_bonus = 2 if any(word in item.doc.text.lower() for word in ["should", "helps", "combines", "grades"]) else 0
        score = item.score + 0.4 * title_bonus + 0.15 * text_overlap + direct_support_bonus
        reranked.append(ScoredDocument(item.doc, score, item.reason + "+support_rerank"))
    return sorted(reranked, key=lambda item: item.score, reverse=True)[:k]


def grade_evidence(query: str, evidence: list[ScoredDocument]) -> str:
    """给当前 top evidence 做粗粒度质量分级。

    Corrective RAG 会根据证据强弱决定是否继续改写 query、补检索或拒答。
    """

    if not evidence:
        return "missing"
    best = evidence[0]
    overlap = lexical_overlap(query, best.doc)
    if overlap >= 3:
        return "strong"
    if overlap >= 1:
        return "partial"
    return "weak"


def rewrite_query(query: str) -> str:
    """把用户 query 改写成更适合检索的形式。

    这里用固定规则模拟 query rewrite / HyDE-like expansion。真实系统可以
    用 LLM 生成多个查询或假想答案，再进入检索阶段。
    """

    replacements = {
        "bad retrieval": "weak irrelevant evidence corrective rag rewrite query refuse answer",
        "not enough evidence": "weak evidence corrective rag retrieve again refuse answer",
        "relationship": "graph entities relationships communities graphrag",
        "whole document": "hierarchical summaries raptor long document retrieval",
    }
    lowered = query.lower()
    additions = [value for key, value in replacements.items() if key in lowered]
    if additions:
        return query + " " + " ".join(additions)
    return query + " retrieval evidence support"


def grounded_answer(question: str, evidence: list[ScoredDocument]) -> str:
    """根据证据生成一个带引用的模板化回答。

    真实系统会把这一步替换成 LLM 调用，但仍然应该保留“只基于证据回答”
    和“输出引用”的约束。
    """

    if not evidence:
        return "资料不足：没有检索到可用证据。"
    lines = [f"问题：{question}", "回答："]
    for idx, item in enumerate(evidence[:2], start=1):
        lines.append(f"{idx}. {item.doc.text} [{item.doc.id}]")
    lines.append("引用：" + ", ".join(f"{item.doc.source}#{item.doc.id}" for item in evidence[:2]))
    return "\n".join(lines)


def decompose_question(question: str) -> list[str]:
    """把复杂问题拆成多个子问题。

    这是 multi-hop RAG 的最小形态：先针对子问题分别检索，再合并证据。
    """

    lowered = question.lower()
    if "compare" in lowered or "对比" in lowered:
        return ["What is sparse BM25 retrieval?", "What is dense embedding retrieval?", question]
    if "production" in lowered or "生产" in lowered:
        return ["How should RAG be evaluated?", "What security risks matter in RAG?", question]
    return [question]


def graph_neighborhood(start: str, max_depth: int = 2) -> list[str]:
    """从图中的起点节点向外扩展邻域。

    GraphRAG 的真实版本会构建实体-关系图和社区摘要。这里用一个小图
    演示“先扩展相关实体，再把图邻域注入检索 query”的思路。
    """

    visited = {start}
    queue = deque([(start, 0)])
    order = []
    while queue:
        node, depth = queue.popleft()
        order.append(node)
        if depth == max_depth:
            continue
        for neighbor in GRAPH_EDGES.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
    return order


def recall_at_k(results: list[ScoredDocument], relevant: set[str], k: int) -> float:
    """计算 Recall@k：top-k 中命中的相关文档占全部相关文档的比例。"""

    found = {item.doc.id for item in results[:k]}
    return len(found & relevant) / len(relevant)


def mrr(results: list[ScoredDocument], relevant: set[str]) -> float:
    """计算 MRR：第一个相关文档排名的倒数。"""

    for rank, item in enumerate(results, start=1):
        if item.doc.id in relevant:
            return 1 / rank
    return 0.0


def print_ranking(title: str, ranking: list[ScoredDocument]) -> None:
    """打印排名结果，帮助观察检索和重排 trace。"""

    print(title)
    for idx, item in enumerate(ranking, start=1):
        print(f"{idx}. {item.doc.id} score={item.score:.3f} reason={item.reason} title={item.doc.title}")
