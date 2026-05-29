from rag_core import BM25Index, CORPUS, grounded_answer, print_ranking, rerank_for_support


question = "Which chunk directly supports why reranking is useful?"
initial = BM25Index(CORPUS).search(question, k=6)
reranked = rerank_for_support(question, initial, k=4)

print_ranking("Initial retrieval:", initial)
print()
print_ranking("Support-aware reranking:", reranked)
print()
print(grounded_answer(question, reranked))

