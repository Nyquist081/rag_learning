from rag_core import CORPUS, dense_style_search, print_ranking, rewrite_query


question = "What should the system do with bad retrieval?"
rewritten = rewrite_query(question)

print(f"Original query: {question}")
print(f"Rewritten/HyDE-like query: {rewritten}")
print()

print_ranking("Dense-style retrieval with original query:", dense_style_search(question, CORPUS, k=4))
print()
print_ranking("Dense-style retrieval with rewritten query:", dense_style_search(rewritten, CORPUS, k=4))

