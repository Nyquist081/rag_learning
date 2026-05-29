from rag_core import BM25Index, CORPUS, dense_style_search, mrr, recall_at_k, reciprocal_rank_fusion, rerank_for_support


EVAL_SET = [
    ("What is sparse retrieval good at?", {"d03"}),
    ("How do embeddings help retrieval?", {"d04"}),
    ("How should weak evidence be handled?", {"d08"}),
    ("Which method uses entities and relationships?", {"d10"}),
    ("What metrics evaluate RAG?", {"d12"}),
]


index = BM25Index(CORPUS)
recalls = []
rrs = []

for question, relevant in EVAL_SET:
    bm25 = index.search(question, k=5)
    dense = dense_style_search(question, k=5)
    hybrid = reciprocal_rank_fusion([bm25, dense], limit=6)
    reranked = rerank_for_support(question, hybrid, k=5)
    recall = recall_at_k(reranked, relevant, k=3)
    reciprocal_rank = mrr(reranked, relevant)
    recalls.append(recall)
    rrs.append(reciprocal_rank)
    print(f"q={question!r} top={[item.doc.id for item in reranked[:3]]} recall@3={recall:.2f} mrr={reciprocal_rank:.2f}")

print(f"\nAverage recall@3={sum(recalls) / len(recalls):.2f}")
print(f"MRR={sum(rrs) / len(rrs):.2f}")

