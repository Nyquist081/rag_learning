from rag_core import BM25Index, CORPUS, graph_neighborhood, grounded_answer, print_ranking, rerank_for_support


question = "How does GraphRAG answer relationship-heavy questions?"
graph_nodes = graph_neighborhood("GraphRAG", max_depth=2)
graph_query = question + " " + " ".join(graph_nodes)

print("Graph neighborhood:")
for node in graph_nodes:
    print(f"- {node}")

evidence = rerank_for_support(graph_query, BM25Index(CORPUS).search(graph_query, k=6), k=4)
print()
print_ranking("Graph-expanded retrieval:", evidence)
print()
print(grounded_answer(question, evidence))

