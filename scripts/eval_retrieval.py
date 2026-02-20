#!/usr/bin/env python3
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _next_archive_path(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{timestamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        archived = _next_archive_path(path)
        path.replace(archived)
        print(f"[archive] moved existing file to {archived}")
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: float) -> tuple[int, dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
            latency_ms = (time.perf_counter() - start) * 1000
            return response.status, json.loads(raw), latency_ms
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        latency_ms = (time.perf_counter() - start) * 1000
        try:
            return exc.code, json.loads(raw), latency_ms
        except Exception:
            return exc.code, {"error": {"message": raw}}, latency_ms
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return 0, {"error": {"message": str(exc), "type": exc.__class__.__name__}}, latency_ms


def _safe_int_ids(values: list[Any]) -> list[int]:
    output: list[int] = []
    for value in values:
        try:
            output.append(int(value))
        except Exception:
            continue
    return output


def _dcg_at_k(ids: list[int], relevance: dict[int, float], k: int) -> float:
    score = 0.0
    for idx, movie_id in enumerate(ids[:k], start=1):
        rel = relevance.get(movie_id, 0.0)
        if rel <= 0:
            continue
        score += (2**rel - 1.0) / math.log2(idx + 1.0)
    return score


def _idcg_at_k(relevance_values: list[float], k: int) -> float:
    sorted_values = sorted([v for v in relevance_values if v > 0], reverse=True)
    score = 0.0
    for idx, rel in enumerate(sorted_values[:k], start=1):
        score += (2**rel - 1.0) / math.log2(idx + 1.0)
    return score


@dataclass
class QueryMetrics:
    query_id: str
    ok: bool
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    title_hit_at_1: float
    expected_no_results: bool | None
    predicted_no_results: bool
    latency_total_ms: float
    latency_retrieval_ms: float | None
    latency_summary_ms: float | None


def run_queries(args: argparse.Namespace) -> int:
    eval_payload = _load_json(Path(args.eval_file))
    queries = eval_payload.get("queries", [])
    records: list[dict[str, Any]] = []
    endpoint = f"{args.base_url.rstrip('/')}{args.query_path}"

    for item in queries:
        query_id = str(item.get("query_id") or "")
        request_payload: dict[str, Any] = {
            "query": item["query"],
            "top_k": int(item.get("top_k", args.default_top_k)),
            "sort": item.get("sort", "relevance"),
        }
        if item.get("filters") is not None:
            request_payload["filters"] = item["filters"]

        status_code, response_json, latency_ms = _post_json(endpoint, request_payload, args.timeout_seconds)
        records.append(
            {
                "query_id": query_id,
                "query": item["query"],
                "request": request_payload,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 2),
                "response": response_json,
            }
        )
        print(f"[run] {query_id or '<no-id>'}: status={status_code} latency_ms={latency_ms:.1f}")

    output = {
        "generated_at": _utc_now(),
        "base_url": args.base_url,
        "query_path": args.query_path,
        "run_name": args.run_name,
        "records": records,
    }
    _write_json(Path(args.out), output)
    print(f"[run] wrote {args.out}")
    return 0


def _query_metrics(eval_query: dict[str, Any], run_record: dict[str, Any], k: int) -> QueryMetrics:
    query_id = str(eval_query.get("query_id", ""))
    response = run_record.get("response") or {}
    status_code = int(run_record.get("status_code", 0))
    ok = status_code == 200

    result_items = response.get("results") if isinstance(response, dict) else None
    if not isinstance(result_items, list):
        result_items = []
    predicted_ids = _safe_int_ids([item.get("id") for item in result_items if isinstance(item, dict)])

    relevant_ids = set(_safe_int_ids(eval_query.get("relevant_ids", [])))
    relevance_by_id_raw = eval_query.get("relevance_by_id") or {}
    relevance_by_id: dict[int, float] = {}
    if isinstance(relevance_by_id_raw, dict):
        for movie_id, rel in relevance_by_id_raw.items():
            try:
                relevance_by_id[int(movie_id)] = float(rel)
            except Exception:
                continue
    if not relevance_by_id:
        relevance_by_id = {movie_id: 1.0 for movie_id in relevant_ids}

    hits = [movie_id for movie_id in predicted_ids[:k] if movie_id in relevant_ids]
    recall = (len(set(hits)) / len(relevant_ids)) if relevant_ids else 0.0

    mrr = 0.0
    for rank, movie_id in enumerate(predicted_ids[:k], start=1):
        if movie_id in relevant_ids:
            mrr = 1.0 / rank
            break

    dcg = _dcg_at_k(predicted_ids, relevance_by_id, k)
    idcg = _idcg_at_k(list(relevance_by_id.values()), k)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    query_type = str(eval_query.get("query_type", "")).lower()
    title_hit = 0.0
    if query_type == "title" and predicted_ids:
        title_hit = 1.0 if predicted_ids[0] in relevant_ids else 0.0

    expected_no_results = eval_query.get("expect_no_results")
    predicted_no_results = bool((response.get("meta") or {}).get("no_relevant_results", False)) or len(predicted_ids) == 0
    meta = response.get("meta") if isinstance(response, dict) else {}
    if not isinstance(meta, dict):
        meta = {}
    breakdown = meta.get("latency_breakdown")
    if not isinstance(breakdown, dict):
        breakdown = {}

    def _maybe_float(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    latency_total = _maybe_float(breakdown.get("total_ms"))
    if latency_total is None:
        latency_total = _maybe_float(meta.get("latency_ms"))
    if latency_total is None:
        latency_total = float(run_record.get("latency_ms", 0.0))

    latency_retrieval = _maybe_float(breakdown.get("retrieval_ms"))
    latency_summary = _maybe_float(breakdown.get("summary_ms"))

    return QueryMetrics(
        query_id=query_id,
        ok=ok,
        recall_at_k=recall,
        mrr_at_k=mrr,
        ndcg_at_k=ndcg,
        title_hit_at_1=title_hit,
        expected_no_results=expected_no_results if isinstance(expected_no_results, bool) else None,
        predicted_no_results=predicted_no_results,
        latency_total_ms=latency_total,
        latency_retrieval_ms=latency_retrieval,
        latency_summary_ms=latency_summary,
    )


def score_run(args: argparse.Namespace) -> int:
    eval_payload = _load_json(Path(args.eval_file))
    run_payload = _load_json(Path(args.run_file))

    eval_queries = eval_payload.get("queries", [])
    records = run_payload.get("records", [])
    record_by_id = {str(item.get("query_id", "")): item for item in records}
    k = args.k

    per_query: list[dict[str, Any]] = []
    metrics: list[QueryMetrics] = []

    for eval_query in eval_queries:
        query_id = str(eval_query.get("query_id", ""))
        run_record = record_by_id.get(query_id)
        if run_record is None:
            run_record = {"query_id": query_id, "status_code": 0, "latency_ms": 0.0, "response": {"error": "missing"}}
        qm = _query_metrics(eval_query, run_record, k)
        metrics.append(qm)
        per_query.append(
            {
                "query_id": qm.query_id,
                "ok": qm.ok,
                f"recall@{k}": round(qm.recall_at_k, 6),
                f"mrr@{k}": round(qm.mrr_at_k, 6),
                f"ndcg@{k}": round(qm.ndcg_at_k, 6),
                "title_hit@1": round(qm.title_hit_at_1, 6),
                "expected_no_results": qm.expected_no_results,
                "predicted_no_results": qm.predicted_no_results,
                "latency_total_ms": round(qm.latency_total_ms, 2),
                "latency_retrieval_ms": round(qm.latency_retrieval_ms, 2) if qm.latency_retrieval_ms is not None else None,
                "latency_summary_ms": round(qm.latency_summary_ms, 2) if qm.latency_summary_ms is not None else None,
            }
        )

    def _avg(values: list[float]) -> float:
        return mean(values) if values else 0.0

    recalls = [m.recall_at_k for m in metrics]
    mrrs = [m.mrr_at_k for m in metrics]
    ndcgs = [m.ndcg_at_k for m in metrics]
    title_hits = [m.title_hit_at_1 for m in metrics if m.title_hit_at_1 >= 0]
    latencies_total = [m.latency_total_ms for m in metrics if m.latency_total_ms > 0]
    latencies_retrieval = [m.latency_retrieval_ms for m in metrics if m.latency_retrieval_ms is not None and m.latency_retrieval_ms > 0]
    latencies_summary = [m.latency_summary_ms for m in metrics if m.latency_summary_ms is not None and m.latency_summary_ms > 0]

    no_result_eval = [m for m in metrics if m.expected_no_results is not None]
    tp = sum(1 for m in no_result_eval if m.expected_no_results and m.predicted_no_results)
    tn = sum(1 for m in no_result_eval if (m.expected_no_results is False) and (m.predicted_no_results is False))
    fp = sum(1 for m in no_result_eval if (m.expected_no_results is False) and m.predicted_no_results)
    fn = sum(1 for m in no_result_eval if m.expected_no_results and (m.predicted_no_results is False))
    no_result_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    no_result_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    no_result_f1 = (
        2 * no_result_precision * no_result_recall / (no_result_precision + no_result_recall)
        if (no_result_precision + no_result_recall) > 0
        else 0.0
    )

    aggregate = {
        "queries_total": len(metrics),
        "queries_ok": sum(1 for m in metrics if m.ok),
        f"avg_recall@{k}": round(_avg(recalls), 6),
        f"avg_mrr@{k}": round(_avg(mrrs), 6),
        f"avg_ndcg@{k}": round(_avg(ndcgs), 6),
        "avg_title_hit@1": round(_avg(title_hits), 6),
        "avg_latency_ms": round(_avg(latencies_total), 2),
        "avg_latency_total_ms": round(_avg(latencies_total), 2),
        "avg_retrieval_latency_ms": round(_avg(latencies_retrieval), 2),
        "avg_summary_latency_ms": round(_avg(latencies_summary), 2),
        "no_result_precision": round(no_result_precision, 6),
        "no_result_recall": round(no_result_recall, 6),
        "no_result_f1": round(no_result_f1, 6),
        "no_result_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }

    output = {
        "generated_at": _utc_now(),
        "k": k,
        "run_file": args.run_file,
        "eval_file": args.eval_file,
        "aggregate": aggregate,
        "per_query": per_query,
    }
    _write_json(Path(args.out), output)

    print(json.dumps(aggregate, indent=2))
    print(f"[score] wrote {args.out}")
    return 0


def compare_scores(args: argparse.Namespace) -> int:
    baseline = _load_json(Path(args.baseline_score))
    candidate = _load_json(Path(args.candidate_score))
    b = baseline.get("aggregate", {})
    c = candidate.get("aggregate", {})

    keys = [
        key
        for key in [
            "queries_ok",
            "avg_recall@5",
            "avg_recall@10",
            "avg_mrr@5",
            "avg_mrr@10",
            "avg_ndcg@5",
            "avg_ndcg@10",
            "avg_title_hit@1",
            "avg_latency_ms",
            "avg_latency_total_ms",
            "avg_retrieval_latency_ms",
            "avg_summary_latency_ms",
            "no_result_precision",
            "no_result_recall",
            "no_result_f1",
        ]
        if key in b and key in c
    ]

    rows: list[dict[str, Any]] = []
    for key in keys:
        b_val = float(b[key])
        c_val = float(c[key])
        rows.append({"metric": key, "baseline": b_val, "candidate": c_val, "delta": round(c_val - b_val, 6)})

    output = {"generated_at": _utc_now(), "baseline_score": args.baseline_score, "candidate_score": args.candidate_score, "rows": rows}
    _write_json(Path(args.out), output)
    print(json.dumps(rows, indent=2))
    print(f"[compare] wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and evaluate retrieval benchmark queries.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute benchmark queries against the API and store raw responses.")
    run_parser.add_argument("--eval-file", required=True, help="Path to labeled query file.")
    run_parser.add_argument("--out", required=True, help="Path to write run output JSON.")
    run_parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL.")
    run_parser.add_argument("--query-path", default="/v1/query", help="Query endpoint path.")
    run_parser.add_argument("--default-top-k", type=int, default=10)
    run_parser.add_argument("--timeout-seconds", type=float, default=60.0)
    run_parser.add_argument("--run-name", default="unnamed")
    run_parser.set_defaults(func=run_queries)

    score_parser = subparsers.add_parser("score", help="Score a run file against labeled relevance data.")
    score_parser.add_argument("--eval-file", required=True)
    score_parser.add_argument("--run-file", required=True)
    score_parser.add_argument("--out", required=True)
    score_parser.add_argument("--k", type=int, default=10)
    score_parser.set_defaults(func=score_run)

    compare_parser = subparsers.add_parser("compare", help="Compare two score JSON files.")
    compare_parser.add_argument("--baseline-score", required=True)
    compare_parser.add_argument("--candidate-score", required=True)
    compare_parser.add_argument("--out", required=True)
    compare_parser.set_defaults(func=compare_scores)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
