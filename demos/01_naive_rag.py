from rag_core import BM25Index, CORPUS, grounded_answer, print_ranking


question = "What is RAG with a retriever, generator, external knowledge base, and evidence?"
index = BM25Index(CORPUS)
evidence = index.search(question, k=3)

print_ranking("Naive BM25 retrieval:", evidence)
print()
print(grounded_answer(question, evidence))
