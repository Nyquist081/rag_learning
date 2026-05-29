from rag_core import BM25Index, CORPUS, decompose_question, grounded_answer, print_ranking, rerank_for_support


question = "Compare sparse and dense retrieval for production RAG."
index = BM25Index(CORPUS)
sub_questions = decompose_question(question)
all_evidence = []

print("Sub-questions:")
for sub_question in sub_questions:
    print(f"- {sub_question}")
    evidence = rerank_for_support(sub_question, index.search(sub_question, k=4), k=2)
    all_evidence.extend(evidence)

deduped = {item.doc.id: item for item in all_evidence}
final_evidence = sorted(deduped.values(), key=lambda item: item.score, reverse=True)

print()
print_ranking("Merged evidence:", final_evidence)
print()
print(grounded_answer(question, final_evidence))

