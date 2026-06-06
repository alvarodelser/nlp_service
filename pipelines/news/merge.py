"""The `merge` orchestrator operation. It lives here, not in any module — it runs on both the
MinHash hit (step 1) and the embedding-dedup match (step 5). Append the new source + url, keep
the oldest date. No re-embedding, no re-summarizing."""


def merge_article(wv, target_id: str, article: dict) -> None:
    cur = wv.get(target_id, return_props=("sources", "urls", "date"))
    sources = list(dict.fromkeys([*cur["sources"], article["source"]]))   # de-dup, keep order
    urls = list(dict.fromkeys([*cur["urls"], article["url"]]))
    oldest = min(cur["date"], article["date"])
    wv.update(target_id, {"sources": sources, "urls": urls, "date": oldest})
