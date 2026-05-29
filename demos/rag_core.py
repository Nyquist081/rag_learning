"""Shared toy components for the chapter demos.

The code is intentionally dependency-free. It models the mechanics of RAG:
chunking, sparse retrieval, dense-style retrieval, fusion, reranking, correction,
multi-hop planning, graph traversal, grounding, and metrics.
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
    id: str
    title: str
    text: str
    source: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScoredDocument:
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
    return [token.lower() for token in TOKEN_RE.findall(text)]


def expanded_terms(text: str) -> list[str]:
    terms = tokenize(text)
    expanded = list(terms)
    for term in terms:
        expanded.extend(RELATED_TERMS.get(term, set()))
    return expanded


def lexical_overlap(query: str, doc: Document) -> int:
    q = set(tokenize(query))
    d = set(tokenize(doc.title + " " + doc.text))
    return len(q & d)


class BM25Index:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.term_freqs = [Counter(tokenize(doc.title + " " + doc.text)) for doc in docs]
        self.doc_lengths = [sum(freq.values()) for freq in self.term_freqs]
        self.avg_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_freqs = Counter()
        for term_freq in self.term_freqs:
            self.doc_freqs.update(term_freq.keys())

    def search(self, query: str, k: int = 5) -> list[ScoredDocument]:
        terms = tokenize(query)
        scored = []
        for idx, doc in enumerate(self.docs):
            score = self._score(terms, idx)
            if score > 0:
                scored.append(ScoredDocument(doc, score, "bm25"))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:k]

    def _score(self, terms: Iterable[str], idx: int, k1: float = 1.5, b: float = 0.75) -> float:
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
    """A tiny embedding-like retriever based on term expansion and cosine similarity."""

    q = Counter(expanded_terms(query))
    scored = []
    for doc in docs:
        d = Counter(expanded_terms(doc.title + " " + doc.text))
        score = cosine(q, d)
        if score > 0:
            scored.append(ScoredDocument(doc, score, "dense_style"))
    return sorted(scored, key=lambda item: item.score, reverse=True)[:k]


def cosine(left: Counter[str], right: Counter[str]) -> float:
    common = set(left) & set(right)
    numerator = sum(left[t] * right[t] for t in common)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def reciprocal_rank_fusion(rankings: list[list[ScoredDocument]], k: int = 60, limit: int = 5) -> list[ScoredDocument]:
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
    reranked = []
    for item in candidates:
        title_bonus = lexical_overlap(query, Document(item.doc.id, item.doc.title, "", item.doc.source))
        text_overlap = lexical_overlap(query, item.doc)
        direct_support_bonus = 2 if any(word in item.doc.text.lower() for word in ["should", "helps", "combines", "grades"]) else 0
        score = item.score + 0.4 * title_bonus + 0.15 * text_overlap + direct_support_bonus
        reranked.append(ScoredDocument(item.doc, score, item.reason + "+support_rerank"))
    return sorted(reranked, key=lambda item: item.score, reverse=True)[:k]


def grade_evidence(query: str, evidence: list[ScoredDocument]) -> str:
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
    if not evidence:
        return "资料不足：没有检索到可用证据。"
    lines = [f"问题：{question}", "回答："]
    for idx, item in enumerate(evidence[:2], start=1):
        lines.append(f"{idx}. {item.doc.text} [{item.doc.id}]")
    lines.append("引用：" + ", ".join(f"{item.doc.source}#{item.doc.id}" for item in evidence[:2]))
    return "\n".join(lines)


def decompose_question(question: str) -> list[str]:
    lowered = question.lower()
    if "compare" in lowered or "对比" in lowered:
        return ["What is sparse BM25 retrieval?", "What is dense embedding retrieval?", question]
    if "production" in lowered or "生产" in lowered:
        return ["How should RAG be evaluated?", "What security risks matter in RAG?", question]
    return [question]


def graph_neighborhood(start: str, max_depth: int = 2) -> list[str]:
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
    found = {item.doc.id for item in results[:k]}
    return len(found & relevant) / len(relevant)


def mrr(results: list[ScoredDocument], relevant: set[str]) -> float:
    for rank, item in enumerate(results, start=1):
        if item.doc.id in relevant:
            return 1 / rank
    return 0.0


def print_ranking(title: str, ranking: list[ScoredDocument]) -> None:
    print(title)
    for idx, item in enumerate(ranking, start=1):
        print(f"{idx}. {item.doc.id} score={item.score:.3f} reason={item.reason} title={item.doc.title}")

