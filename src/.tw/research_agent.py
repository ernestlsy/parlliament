"""
Research Agent -- pulls real evidence from arXiv to ground the Evolution
Judge's hypotheses in actual literature instead of a hardcoded briefing.
Free, keyless API. Results are cached to disk per topic (papers don't change,
and arXiv asks callers not to hammer its API), so a whole run costs at most
one live fetch per topic.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "research_cache.json")
ARXIV_NS = {"a": "http://www.w3.org/2005/Atom"}

# One topic per untested headroom direction from the starter kit README,
# rotated across experiment ids so a run accumulates evidence on different
# fronts instead of re-querying the same thing every iteration.
TOPICS = [
    ("pairwise_ranking_loss", "pairwise ranking loss BPR recommendation"),
    ("sequence_modeling", "user behavior sequence modeling recommendation interest network"),
    ("multi_task_learning", "multi-task learning recommendation auxiliary click view"),
    ("watch_time_debiasing", "watch time censored regression recommendation debiasing"),
    ("factorization_machine", "factorization machine deep learning ranking recommendation"),
]


def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def _save_cache(cache: dict):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _fetch_arxiv(query: str, max_results: int = 4) -> list:
    params = urllib.parse.urlencode({
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    req = urllib.request.Request(
        f"http://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": "kuairand-research-agent/1.0 (hackathon project)"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        root = ET.fromstring(resp.read())
    papers = []
    for entry in root.findall("a:entry", ARXIV_NS):
        title = entry.find("a:title", ARXIV_NS).text.strip().replace("\n", " ")
        summary = entry.find("a:summary", ARXIV_NS).text.strip().replace("\n", " ")
        published = entry.find("a:published", ARXIV_NS).text[:4]
        link = entry.find("a:id", ARXIV_NS).text
        papers.append({"title": title, "year": published, "summary": summary[:400], "url": link})
    return papers


def get_findings(topic_index: int) -> dict:
    """Fetch (or reuse cached) papers for one rotating research topic.
    Never raises -- on any failure it returns an empty paper list so the
    Evolution Judge just proceeds without that iteration's research input."""
    label, query = TOPICS[topic_index % len(TOPICS)]
    cache = _load_cache()
    if label in cache:
        return cache[label]

    result = {"topic": label, "query": query, "papers": [], "error": None}
    for attempt in range(2):
        try:
            result["papers"] = _fetch_arxiv(query)
            break
        except Exception as e:
            result["error"] = str(e)
            if attempt == 0:
                time.sleep(3)

    if result["papers"]:  # only cache real hits, so a transient failure can retry next run
        cache[label] = result
        _save_cache(cache)
    return result


def format_findings(findings: dict) -> str:
    if not findings["papers"]:
        note = f" ({findings['error']})" if findings.get("error") else ""
        return f"[Research Agent] No papers retrieved for topic '{findings['topic']}'{note}."
    lines = [f"Research findings -- topic: {findings['topic']} "
             f"(live arXiv query: \"{findings['query']}\")"]
    for p in findings["papers"]:
        lines.append(f"- {p['title']} ({p['year']}): {p['summary']}")
    return "\n".join(lines)
