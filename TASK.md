## Project Scope: Email Social Network Analysis (IIMB Org Design)

### Mandatory Deliverables (Phase 1)
1. Corpus
- Use Enron email corpus (preferred) or another public organizational corpus.

2. Metadata extraction (header only)
- Extract: `Date`, `From`, `To`, `Cc`, `Subject`.
- Do not require email body for Phase 1.

3. Network construction
- Node = person (email ID).
- Edge = communication tie between sender and recipient.

4. Tie-strength weighting
- For one email sent to `N` recipients (`To + Cc`), assign weight `1/N` from sender to each recipient.
- Sum these over all emails to get final edge strength.

5. Interactive visualization
- Force-directed network view.
- Community/clustering view (density-based grouping).
- Edge-strength threshold slider on logarithmic scale.

### Optional / Phase 2 (Do Later)
- Infer likely departments (HR/Finance/etc.) from communication patterns + subject tokens.
- Infer relative influence/power using network metrics (e.g., PageRank, betweenness, broker behavior).

### Out of Scope for Phase 1 (avoid overkill)
- Full NLP/content modeling of bodies.
- Causal claims about hierarchy without validation labels.
- Heavy predictive modeling before baseline SNA is finalized.

---
