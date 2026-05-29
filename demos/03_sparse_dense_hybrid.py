from rag_core import BM25Index, CORPUS, dense_style_search, print_ranking, reciprocal_rank_fusion


question = "How can semantic retrieval and keyword retrieval work together?"
bm25 = BM25Index(CORPUS).search(question, k=5)
dense = dense_style_search(question, k=5)
hybrid = reciprocal_rank_fusion([bm25, dense], limit=5)

print_ranking("BM25:", bm25)
print()
print_ranking("Dense-style:", dense)
print()
print_ranking("Hybrid RRF:", hybrid)

