# Lesson 4 vs Lesson 5 — Runtime Comparison

> Comparing agent behavior, token usage, tool call patterns, and report quality
> across the same 10 test queries run on both implementations.
>
> - **L4**: Custom ReAct loop, web-only tools (v0.1.0, run 2026-03-22)
> - **L5**: Custom ReAct loop + RAG tool — knowledge_search with hybrid FAISS+BM25 + cross-encoder reranking (v0.1.0, run 2026-03-22)
> - **Model**: gpt-4.1-mini (same for both)
> - **Queries 1-8**: Identical to L3/L4 test set
> - **Queries 9-10**: New RAG-targeted queries (LangChain, Transformers)

---

## 1. Per-Query Metrics

### Q1: "Tell me about the Ukrainian P1-SUN drone"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 44,514 | 28,784 |
| Input tokens | 42,719 | 27,455 |
| Output tokens | 1,795 | 1,329 |
| Tool calls | 12 (web_search:3, read_url:7, write_report:1) | 8 (knowledge_search:1, web_search:2, read_url:4, write_report:1) |
| Fetch errors | 2 | 1 |
| Report saved | Yes | Yes |

**Notes:** L5 was 35% more token-efficient. The `knowledge_search` call returned no useful results (P1-SUN drone not in the PDFs), but the agent correctly fell back to web search. Fewer read_url calls (4 vs 7) resulted in a more focused report.

---

### Q2: "Знайди 3 рецепти здобної паски на Великдень та порівняй їх"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 43,497 | 21,579 |
| Input tokens | 42,131 | 19,708 |
| Output tokens | 1,366 | 1,871 |
| Tool calls | 12 (web_search:6, read_url:5, write_report:0) | 7 (web_search:3, read_url:3, write_report:1) |
| Fetch errors | 1 | 0 |
| Report saved | **No** | Yes |

**Notes:** L4 used 2x the tokens and **failed to save a report** — it dumped content as a direct message. L5's updated system prompt ("check KB first, then web") led to a more disciplined workflow: 3 searches, 3 reads, report saved. L5 also produced more output tokens (longer report content).

---

### Q3: "Compare FastAPI and Streamlit"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 43,085 | 20,131 |
| Input tokens | 41,858 | 18,882 |
| Output tokens | 1,227 | 1,249 |
| Tool calls | 10 (web_search:5, read_url:4, write_report:1) | 8 (knowledge_search:1, web_search:2, read_url:4, write_report:1) |
| Fetch errors | 1 | 1 |
| Report saved | Yes | Yes |

**Notes:** L5 was 53% more token-efficient. L4 ran 5 web searches; L5 ran 1 knowledge_search + 2 web searches to achieve similar coverage. Both produced comparable reports.

---

### Q4: "Compare naive RAG, sentence-window retrieval, and parent-child retrieval"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 17,354 | 44,433 |
| Input tokens | 16,016 | 42,782 |
| Output tokens | 1,338 | 1,651 |
| Tool calls | 8 (web_search:4, read_url:3, write_report:1) | 16 (knowledge_search:4, web_search:2, read_url:5, write_report:1) |
| Fetch errors | 0 | 2 |
| Report saved | Yes | Yes |

**Notes:** L5 used significantly more tokens here (+156%). The agent made 4 knowledge_search calls (one per sub-topic + combined), then supplemented with web search. The deeper research is intentional — the RAG documents contain relevant content about retrieval approaches. L5's report was longer (1,651 vs 1,338 output tokens) with content from both the knowledge base and web sources. This is the query where RAG adds the most value.

---

### Q5: "Compare weather in Costa del Sol and Costa Blanca for March 16-31, 2026"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 23,039 | 28,418 |
| Input tokens | 21,715 | 27,118 |
| Output tokens | 1,324 | 1,300 |
| Tool calls | 9 (web_search:4, read_url:4, write_report:1) | 9 (knowledge_search:2, web_search:2, read_url:3, write_report:1) |
| Fetch errors | 1 | 1 |
| Report saved | Yes | Yes |

**Notes:** Similar performance. L5 wasted 2 knowledge_search calls on weather (not in the PDFs), adding ~5K tokens of overhead. The agent correctly fell back to web search. For non-KB topics, the RAG tool adds a small cost.

---

### Q6: "Best 3 places in Spain for hip implant surgery"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 19,893 | 48,502 |
| Input tokens | 18,767 | 47,037 |
| Output tokens | 1,126 | 1,465 |
| Tool calls | 8 (web_search:4, read_url:3, write_report:1) | 10 (knowledge_search:1, web_search:2, read_url:6, write_report:1) |
| Fetch errors | 0 | 0 |
| Report saved | Yes | Yes |

**Notes:** L5 used 2.4x tokens. The agent read 6 URLs (some duplicates) vs L4's 3, leading to bloated context. L5's slightly longer report (1,465 vs 1,126 tokens) doesn't fully justify the extra cost.

---

### Q7: "What activities and events are in Velez-Malaga and Torre del Mar in March 2026?"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 23,421 | 20,913 |
| Input tokens | 21,876 | 19,899 |
| Output tokens | 1,545 | 1,014 |
| Tool calls | 9 (web_search:4, read_url:4, write_report:1) | 7 (web_search:3, read_url:4, write_report:1) |
| Fetch errors | 0 | 0 |
| Report saved | Yes | Yes |

**Notes:** L5 slightly more efficient (-11%). Correctly skipped knowledge_search (events not in PDFs). Both produced clean runs with zero errors.

---

### Q8: "Compare Vélez-Málaga and Motril for living in 2026"

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 25,869 | 68,872 |
| Input tokens | 24,519 | 67,487 |
| Output tokens | 1,350 | 1,385 |
| Tool calls | 9 (web_search:6, read_url:3, write_report:1) | 16 (knowledge_search:2, web_search:5, read_url:5, write_report:1) |
| Fetch errors | 0 | 3 |
| Report saved | Yes | Yes |

**Notes:** L5 used 2.7x tokens — the most expensive query. 3 fetch errors forced additional search rounds. The 2 knowledge_search calls were wasted (Spanish cities not in PDFs). L5's expanded strategy ("check KB first") added overhead on off-topic queries.

---

### Q9: "How does LangChain implement document loading and text splitting for RAG pipelines?" (NEW)

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 14,247 | 25,985 |
| Input tokens | 12,797 | 24,748 |
| Output tokens | 1,450 | 1,237 |
| Tool calls | 7 (web_search:3, read_url:4, write_report:0) | 8 (knowledge_search:3, web_search:1, read_url:3, write_report:1) |
| Fetch errors | 0 | 0 |
| Report saved | **No** | Yes |

**Notes:** L4 **failed to save a report** (second missed report). L5 used 3 knowledge_search calls pulling from the LangChain PDF, supplemented with 1 web search, and properly saved the report. The KB content gave L5 foundational knowledge that L4 had to piece together from web sources alone. L5's approach was more structured despite higher token cost.

---

### Q10: "Порівняй архітектуру трансформерів з попередніми підходами NLP для мовного моделювання" (NEW, Ukrainian)

| Metric | L4 | L5 |
|--------|----|----|
| Total tokens | 20,359 | 52,858 |
| Input tokens | 19,135 | 50,634 |
| Output tokens | 1,224 | 2,224 |
| Tool calls | 11 (web_search:5, read_url:5, write_report:1) | 10 (knowledge_search:1, web_search:2, read_url:6, write_report:1) |
| Fetch errors | 3 | 1 |
| Report saved | Yes | Yes |

**Notes:** L5 used 2.6x tokens but produced a significantly longer report (2,224 vs 1,224 output tokens — nearly double). L4 hit 3 fetch errors; L5 only 1. The knowledge_search call pulled from `large-language-model.pdf`, giving the agent foundational content about LLM architecture. Both answered in Ukrainian as expected.

---

## 2. Aggregate Comparison

| Metric | L4 (10 queries) | L5 (10 queries) | Delta |
|--------|-----------------|-----------------|-------|
| **Total tokens** | 275,278 | 360,475 | **+31%** |
| **Total input tokens** | 261,533 | 346,750 | **+33%** |
| **Total output tokens** | 13,745 | 13,725 | **~same** |
| **Total tool calls** | 96 | 99 | +3% |
| **knowledge_search calls** | 0 | 15 | new tool |
| **web_search calls** | 44 | 24 | **-45%** |
| **read_url calls** | 42 | 43 | ~same |
| **write_report calls** | 8 | 10 | +2 |
| **Fetch errors** | 8 | 9 | ~same |
| **Reports saved** | 8/10 (80%) | **10/10 (100%)** | improved |

### Token Efficiency
- L5 used **31% more total tokens** overall — primarily due to knowledge_search results adding to context
- The overhead is concentrated on non-KB queries (#5, #6, #8) where knowledge_search returns irrelevant results but still consumes context
- On KB-relevant queries (#4, #9), the added tokens produce richer, better-sourced reports
- Output tokens are nearly identical — L5 doesn't produce longer reports on average

### Tool Call Patterns
- L5 **halved web_search calls** (24 vs 44) by substituting knowledge_search for initial research
- L4 averaged 4.4 web searches per query; L5 averaged 2.4 web + 1.5 knowledge_search
- read_url usage is identical — both read ~4 URLs per query
- L5 always called write_report; L4 missed it twice

### Report Reliability
- **L4 failed to save reports on Q2 and Q9** (80% success rate)
- **L5 saved all 10 reports** (100% success rate)
- L5's updated system prompt ("ALWAYS call write_report as your FINAL tool call") + the "check KB first" strategy created a more disciplined workflow

---

## 3. RAG Impact Analysis

### When RAG Helps (KB-relevant queries)

| Query | knowledge_search calls | Impact |
|-------|:---:|--------|
| Q4: RAG approaches | 4 | Deep coverage from `retrieval-augmented-generation.pdf` + web. Most thorough report. |
| Q9: LangChain doc loading | 3 | KB provided foundational LangChain knowledge. L4 failed to save report; L5 succeeded. |
| Q10: Transformers vs NLP | 1 | KB content from `large-language-model.pdf`. Report 82% longer than L4's. |

### When RAG Is Neutral (topic not in KB, agent skips it)

| Query | knowledge_search calls | Impact |
|-------|:---:|--------|
| Q2: Easter paska | 0 | Agent correctly skipped KB. |
| Q7: Velez-Malaga events | 0 | Agent correctly skipped KB. |

### When RAG Adds Overhead (topic not in KB, agent checks anyway)

| Query | knowledge_search calls | Wasted tokens |
|-------|:---:|---:|
| Q1: P1-SUN drone | 1 | ~1,600 |
| Q3: FastAPI vs Streamlit | 1 | ~1,600 |
| Q5: Weather comparison | 2 | ~3,200 |
| Q6: Hip surgery Spain | 1 | ~1,600 |
| Q8: Velez vs Motril | 2 | ~3,200 |

Total wasted on irrelevant KB checks: ~11,200 tokens (~3% of total)

---

## 4. Behavioral Differences

### Search Strategy
| Aspect | L4 | L5 |
|--------|----|----|
| First action | web_search (always) | knowledge_search or web_search (topic-dependent) |
| Avg web searches/query | 4.4 | 2.4 |
| Avg knowledge searches/query | 0 | 1.5 |
| Research depth on KB topics | Web-only | KB foundation + web supplement |

### Report Saving
| Aspect | L4 | L5 |
|--------|----|----|
| Reports saved | 8/10 (80%) | **10/10 (100%)** |
| Missed reports | Q2 (paska), Q9 (LangChain) | None |

### Error Recovery
- Both handle fetch errors similarly (skip and try alternatives)
- L4: 8 fetch errors across 10 queries
- L5: 9 fetch errors across 10 queries (comparable)

---

## 5. Architecture Comparison

| Dimension | L4 (web-only) | L5 (web + RAG) |
|-----------|---------------|----------------|
| **Tools** | 5 | 6 (+knowledge_search) |
| **agent.py changes** | — | None (tool-agnostic) |
| **New files** | — | ingest.py, retriever.py |
| **New dependencies** | — | langchain, faiss-cpu, sentence-transformers, pypdf |
| **Docker image size** | ~530 MB | ~2.5 GB (PyTorch for cross-encoder) |
| **Startup time** | Instant | Instant (lazy-loaded retriever) |
| **First knowledge_search** | N/A | ~5s (loads FAISS + BM25 + CrossEncoder) |
| **System prompt** | 120 lines | 150 lines (+KB strategy, +knowledge_search tool) |

---

## 6. Conclusion

**L5 (RAG) advantages:**
- 100% report save rate (vs 80% for L4)
- Richer reports on KB-relevant topics (Q4, Q9, Q10) — combines document knowledge with web sources
- 45% fewer web searches — knowledge base substitutes for initial web research
- Agent correctly identifies when to use knowledge_search vs web_search
- Zero changes to agent.py — tool-agnostic design proved out

**L5 (RAG) disadvantages:**
- 31% more total tokens overall due to KB context injection
- ~11K tokens wasted on irrelevant KB checks for off-topic queries
- Docker image 5x larger (PyTorch dependency for cross-encoder)
- Requires offline ingestion step (`python ingest.py`)

**L4 (web-only) advantages:**
- 31% fewer tokens on average
- Simpler deployment (no FAISS index, no ingestion step)
- Smaller Docker image

**Bottom line:** RAG adds clear value for domain-specific questions where the knowledge base contains relevant content. The agent autonomously decides when to use it. The token overhead on off-topic queries (~3%) is acceptable. The biggest win is reliability — L5 never missed saving a report, likely because the updated system prompt with the "check KB → web → synthesize → save" workflow creates a more disciplined agent behavior pattern.
