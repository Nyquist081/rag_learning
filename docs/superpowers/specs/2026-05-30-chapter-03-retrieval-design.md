# Chapter 03 Retrieval Design

Date: 2026-05-30

## Goal

Expand Chapter 03, `Sparse、Dense 与 Hybrid Retrieval`, into a complete tutorial chapter. The chapter should match the depth and style of Chapters 01 and 02: theory first, then implementation principles, engineering trade-offs, failure modes, demo mapping, and practical checklists.

After reading the chapter, the reader should be able to:

- Explain what retrieval is responsible for inside a RAG system.
- Distinguish sparse retrieval, dense retrieval, and hybrid retrieval.
- Understand the BM25 formula at the level needed for engineering decisions.
- Understand embedding retrieval, vector similarity, and why semantic similarity is not the same as answer support.
- Explain why hybrid retrieval is commonly used in production RAG.
- Use RRF and weighted fusion conceptually, including trade-offs.
- Diagnose common retrieval failures before blaming the generator.
- Read and explain `demos/03_sparse_dense_hybrid.py` and the related functions in `demos/rag_core.py`.

## Scope

This design covers only Chapter 03 and its explanation of the existing demo.

In scope:

- Rewrite `chapters/03_retrieval_sparse_dense_hybrid.md`.
- Keep the current demo code unchanged unless a small documentation-driven adjustment is clearly needed.
- Reference current public research and established retrieval concepts.
- Preserve the chapter's role in the course sequence.

Out of scope:

- Deep reranking implementation. Chapter 05 owns reranking.
- Query rewriting and HyDE details. Chapter 04 owns those.
- Full ANN internals. Chapter 02 already introduces indexing; Chapter 03 should only explain enough to place vector search in the retrieval pipeline.
- Production monitoring and safety. Later chapters cover those.

## Recommended Approach

Use a "layered engine room" structure: explain the retrieval engine from first principles, then add sparse retrieval, dense retrieval, hybrid retrieval, tuning, diagnostics, and demo mapping.

This balances theory and engineering practice. It avoids becoming either a pure math note or a production checklist with shallow algorithm explanations.

## Chapter Structure

The expanded chapter should have these sections:

1. **Retrieval's role in RAG**
   - Retrieval is candidate evidence selection, not final answering.
   - The retriever optimizes recall under a context budget.
   - Bad retrieval caps answer quality.

2. **Formalizing retrieval**
   - Define query `q`, corpus `D`, candidate documents `d`, scoring function `s(q, d)`, and `top-k`.
   - Explain recall vs precision for RAG.
   - Explain candidate pool vs final context.

3. **Sparse retrieval**
   - Explain inverted indexes.
   - Explain TF, IDF, document length normalization.
   - Present BM25 formula and intuition.
   - Explain `k1` and `b`.
   - List strengths: exact identifiers, product names, error codes, code symbols.
   - List weaknesses: paraphrases, vocabulary mismatch, multilingual semantic matching.

4. **Dense retrieval**
   - Explain embedding models and vector space.
   - Explain cosine similarity, dot product, and normalization.
   - Explain semantic similarity and paraphrase matching.
   - Explain dense retrieval failure modes: exact token misses, false semantic neighbors, embedding model mismatch, domain drift.

5. **ANN and vector search in context**
   - Briefly explain why brute force vector search is expensive.
   - Place HNSW/IVF/PQ/vector databases in the retrieval pipeline.
   - Avoid repeating Chapter 02's indexing details.

6. **Hybrid retrieval**
   - Explain why production RAG often uses multiple recall paths.
   - Describe candidate union, deduplication, score normalization, weighted fusion, and RRF.
   - Present RRF formula and why it is robust when scores are not comparable.
   - Explain when weighted fusion is useful and risky.

7. **Metadata and permission filters**
   - Explain pre-filter vs post-filter.
   - Explain why filtering after top-k can hurt recall and leak traces.
   - Cover time, document type, tenant, language, and ACL filters.

8. **Failure modes and diagnostics**
   - No relevant evidence in top-k.
   - Relevant evidence retrieved but ranked too low.
   - Dense retrieval misses identifiers.
   - BM25 misses paraphrases.
   - Hybrid returns duplicates or noisy near-matches.
   - Top-k too small loses evidence; top-k too large pollutes generation.

9. **Tuning guide**
   - `top_k`.
   - Candidate pool size before rerank.
   - BM25 `k1` and `b`.
   - Embedding model choice.
   - Similarity function and normalization.
   - RRF constant.
   - Fusion weights.

10. **Demo walkthrough**
    - Map `BM25Index(CORPUS).search(...)` to sparse retrieval.
    - Map `dense_style_search(...)` to simplified dense retrieval.
    - Map `reciprocal_rank_fusion(...)` to hybrid retrieval.
    - Explain why the demo uses term expansion instead of a real embedding model.
    - Explain how to replace the toy dense retriever with a real embedding model later.

11. **Practical checklist**
    - What to log.
    - What metrics to inspect.
    - How to choose sparse, dense, or hybrid for a new corpus.
    - What questions to ask before tuning the generator.

12. **References**
    - BEIR.
    - BM25 / probabilistic relevance framework.
    - Dense Passage Retrieval.
    - ColBERT or late interaction as a bridge concept.
    - RAG Survey.
    - Optional recent retrieval/hybrid RAG references if directly useful.

## Algorithm Depth

BM25 should include the formula:

```text
score(q, d) = sum IDF(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
```

The chapter should explain:

- `f(t,d)` means term frequency.
- `IDF(t)` means term rarity.
- `|d| / avgdl` is length normalization.
- `k1` controls term frequency saturation.
- `b` controls how much document length matters.

Dense retrieval should include:

```text
v_q = embed(q)
v_d = embed(d)
score = cosine(v_q, v_d)
```

Hybrid retrieval should include:

```text
RRF(d) = sum 1 / (k + rank_i(d))
```

The math should support understanding, not turn the chapter into a proof-heavy paper note.

## Demo Requirements

The existing demo should remain simple and dependency-free:

```bash
python demos/03_sparse_dense_hybrid.py
```

The chapter should explicitly say that `dense_style_search` is not a real embedding retriever. It uses term expansion and cosine similarity so the course can demonstrate dense-style behavior without external models or vector databases.

## Acceptance Criteria

- Chapter 03 is expanded to a self-contained tutorial.
- The chapter explains both algorithm principles and engineering trade-offs.
- The chapter clearly distinguishes retrieval, reranking, and generation.
- The chapter includes formulas for BM25, cosine similarity, and RRF.
- The demo walkthrough accurately matches the current code.
- The chapter does not duplicate Chapter 02's deep indexing content.
- The chapter does not move Chapter 04 query rewriting or Chapter 05 reranking into this chapter.
- Existing demo scripts still run.

## Validation

After implementation, run:

```bash
python demos/03_sparse_dense_hybrid.py
python -m py_compile demos/*.py practice/toy_rag.py
```

Check git diff to ensure only intended files changed.

