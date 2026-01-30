# LitScout (Literature Scout) — MVP-1

LitScout is a lightweight research tooling CLI that searches papers from arXiv and Semantic Scholar, normalizes and deduplicates results, and persists them to Markdown, JSONL, and a SQLite database.

## Installation

```bash
pip install -e .
```

## Environment Variables

- `S2_API_KEY`: Semantic Scholar Graph API key. If missing, LitScout will warn and continue using arXiv only.
- `OPENAI_API_KEY`: Required for `litscout enrich` when provider is `openai`.
- `MIMO_API_KEY`: Required for `litscout enrich` when provider is `mimo`.
- `DEEPSEEK_API_KEY`: Required for `litscout enrich` when provider is `deepseek`.
- `XAI_API_KEY`: Required for `litscout enrich` when provider is `grok`.

## CLI Usage

```bash
litscout search "graph neural networks" --topk 5 --since 2020
```

Common options:
- `--query`: Additional search query (repeatable)
- `--queries-file`: File with one query per line
- `--topk`: Max results per source (default: 100)
- `--since`: Filter by year >= since
- `--year-to`: Filter by year <= year-to
- `--out-md`: Output Markdown path (default: `output-<timestamp>/results_raw.md`)
- `--out-jsonl`: Output JSONL path (default: `output-<timestamp>/results_raw.jsonl`)
- `--db`: SQLite database path (default: `~/.litscout/litscout.db`)
- `--no-cache`: Ignore cached results (still writes fresh cache)
- `--max-requests-per-second`: Global request rate limit (default: 0.3333)
- `--keyword-rank`: Rank by keyword hits (title>abstract)
- `--strict`: Disable title-based deduplication (only DOI/arXiv matches)
- `--cache-ttl-seconds`: Cache TTL in seconds (default: 86400)
- `--cache-max-entries`: Max cache entries (default: 2000)
- `--log-level`: Log level (DEBUG, INFO, WARNING, ERROR)

## Output Formats

1) **Markdown (RAW)**
- Human-readable report of normalized papers with source records and dedup notes.

2) **JSONL**
- One canonical paper per line, fully serialized with authors and sources.

3) **SQLite**
- Entity storage with `papers`, `authors`, `paper_authors`, and `sources` tables.
- Includes a simple cache table for API responses.

## MVP-2: Enrichment (AI 解读 + 排序 + 分类)

新增功能要点：
- AI 读元数据生成精简摘要与方法关键词
- 更丰富的检索与排序策略（关键词匹配 + 规则分类 + 可选 goal 打分）
- 评分机制统一到轻量 AI JSON，并在本地完成排序与渲染

Prepare a skill file (examples in `docs/skill.md.example` and `docs/skill.en.md.example`).
You can define multiple research goals in the skill file under `# Research Goals` with `## G1: ...` headings.

```bash
litscout enrich \\
  --goal \"sign problem phase transition\" \\
  --skill docs/skill.md.example \\
  --in-jsonl results_raw.jsonl \\
  --out-jsonl results_enriched.jsonl \\
  --out-md results_enriched.md
```

Common enrich options:
- `--goal`: Research goal (required)
- `--skill`: Path to skill.md (required)
- `--in-jsonl`: Input raw JSONL (default: `results_raw.jsonl`)
- `--out-jsonl`: Output enriched JSONL (default: `results_enriched.jsonl`)
- `--out-md`: Output enriched Markdown (default: `results_enriched.md`)
- `--provider` / `--model` / `--temperature` / `--max-tokens`: Override LLM settings
- `--base-url`: Override provider base URL (useful for OpenAI-compatible endpoints)
- `--use-goals`: Use goal match score in ranking (default: off)
- `--goal-weight`: Weight for goal match when `--use-goals` is set
- `--llm-concurrency`: Max concurrent LLM requests (default: 1, suggest 8-16 for MIMO)
- `--batch-size`: Max papers per LLM request (default: 5)
- `--llm-max-requests-per-second`: LLM request rate limit (default: 16.6667)

Enriched output includes structured JSONL with a lightweight AI insight schema (summary, methods, priority, score, goal_scores), plus ranking, rule-based classification, and provenance metadata.

Lite insight schema (per paper, single-item response or within `items`):
```json
{
  "paper_id": "...",
  "summary": "One-sentence summary",
  "methods": ["DQMC", "AFQMC"],
  "priority": "P2",
  "score": 0.62,
  "goal_scores": [{"goal_id": "G1", "score": 0.45}]
}
```

## Rate Limiting & Retries

- A shared async rate limiter controls requests across arXiv and Semantic Scholar.
- Failed requests are retried up to 2 times with exponential backoff.

## Cache Cleanup

You can manually clean the cache to enforce TTL or size caps:

```bash
litscout cache-cleanup --cache-ttl-seconds 86400 --cache-max-entries 2000
```

## Extending to New Sources

Add a new connector under `src/litscout/connectors/`, normalize results into the
`CanonicalPaper` model, and include it in `cli.py`. The deduper and storage
layers will work with any source that follows the canonical model.
