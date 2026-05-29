from rag_core import BM25Index, CORPUS, grade_evidence, grounded_answer, print_ranking, rerank_for_support, rewrite_query


question = "How should bad retrieval be corrected?"
index = BM25Index(CORPUS)
first_pass = rerank_for_support(question, index.search(question, k=4))
grade = grade_evidence(question, first_pass)

print_ranking("First pass:", first_pass)
print(f"\nEvidence grade: {grade}")

if grade in {"missing", "weak", "partial"}:
    rewritten = rewrite_query(question)
    print(f"Corrective rewrite: {rewritten}\n")
    second_pass = rerank_for_support(rewritten, index.search(rewritten, k=6))
    print_ranking("Second pass:", second_pass)
    print()
    print(grounded_answer(question, second_pass))
else:
    print(grounded_answer(question, first_pass))
