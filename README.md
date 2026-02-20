# Intelahub Retrieval API

this repo is the movie retrieval backend for the assessment

what is inside
- tmdb ingestion for 200 plus movies
- embeddings with openai by default
- chroma hybrid retrieval lexical plus vector
- cross encoder reranker with heuristic fallback
- llm intent parse plus optional query expansion
- deterministic guardrails
- openai summary with gemini fallback

## quick start

1 create and activate a venv
2 install deps

```bash
pip install -e .[dev]
```

3 create env file

```bash
cp .env.example .env
```

4 run api

```bash
uvicorn app.main:app --reload
```

## docker

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f
```

if you only changed env values

```bash
docker compose up -d --force-recreate api
```

if you need to stop it

```bash
docker compose down
```

## endpoints

- `POST /v1/ingest`
- `POST /v1/query`
- `GET /health/live`

`/health/live` is just a simple alive check

## request examples

ingest

```json
{
  "target_count": 250,
  "force_reindex": false
}
```

query

```json
{
  "query": "best action movies after 2018 rating above 7.5",
  "filters": {
    "genres": ["action"],
    "min_year": 2018,
    "min_rating": 7.5
  },
  "top_k": 10,
  "sort": "relevance"
}
```

response has clear score fields so it is easy to debug
- `vector_score`
- `lexical_score`
- `fused_score`
- `cross_encoder_score`
- `rerank_score`
- `no_results_reason`

## guardrails

guardrails are deterministic
- prompt injection patterns
- secret exfiltration patterns
- suspicious token checks
- query length limit
- filter sanity checks

blocked queries return 400 with a clear code

## retrieval rules

- cosine metric in chroma
- relevance threshold gate
- title override for exact name lookups
- no results safety check for open queries
- optional llm query expansion when query is weak and eligible

current recommended settings

```env
RELEVANCE_THRESHOLD=0.38
ENABLE_NO_RESULTS_SAFETY_CHECK=true
NO_RESULTS_MIN_VECTOR_SCORE=0.35
NO_RESULTS_MIN_LEXICAL_SCORE=0.03
RERANKER_MODE=cross_encoder
ENABLE_LLM_QUERY_EXPANSION=true
ENABLE_LLM_QUERY_PARSER=true
LLM_QUERY_EXPANSION_MIN_CONFIDENCE=0.75
LLM_QUERY_EXPANSION_TRIGGER_SCORE=0.55
LLM_QUERY_EXPANSION_MAX_QUERY_TOKENS=12
MAX_QUERY_EXPANSION_TERMS=4
```

if confidence is too low the api returns no results and skips summary

## postman

use this file
- `docs/postman_collection.json`

it has health ingest query guardrail and low confidence examples

## design explanation

the api is split by responsibility
- routes handle request and response
- services handle retrieval ingestion guardrails and llm calls
- vector store handles indexing and search

retrieval is hybrid
- lexical search for title and keyword matching
- vector search for semantic matching
- rerciprocal rank fusion then rerank
- final no results safety check for weak open queries

## scaling notes

- stateless api replicas behind a load balancer
- stateful vector store on persistent storage or managed vector db
- distributed rate limiting via redis or api gateway for multi replica consistency
- async ingestion workers plus queue for bulk refreshes
- structured logs metrics tracing for observability
- cost controls with provider choice embedding batch size top_k caps and caching strategy

## tests

```bash
pytest
```

## eval

prepare file once

```bash
cp data/eval_queries.template.json data/eval_queries.json
```

run baseline

```bash
make eval-baseline BASE_URL=http://localhost:8000 QUERY_PATH=/v1/query
```

run candidate

```bash
make eval-candidate BASE_URL=http://localhost:8000 QUERY_PATH=/v1/query
```

compare

```bash
make eval-compare
```

key metrics to watch
- `avg_recall@10`
- `avg_mrr@10`
- `avg_ndcg@10`
- `avg_latency_total_ms`
- `no_result_precision`
- `no_result_recall`
- `no_result_f1`
