from unittest.mock import MagicMock

from pipelines.news.merge import merge_article


def test_appends_new_source_and_url_keeps_oldest_date():
    wv = MagicMock()
    wv.get.return_value = {
        "sources": ["elpais"],
        "urls": ["http://a"],
        "date": "2026-01-10",
    }
    article = {"source": "elmundo", "url": "http://b", "date": "2026-01-05"}

    merge_article(wv, "obj-1", article)

    wv.update.assert_called_once_with("obj-1", {
        "sources": ["elpais", "elmundo"],
        "urls": ["http://a", "http://b"],
        "date": "2026-01-05",
    })


def test_does_not_duplicate_existing_source_or_url():
    wv = MagicMock()
    wv.get.return_value = {
        "sources": ["elpais"],
        "urls": ["http://a"],
        "date": "2026-01-01",
    }
    article = {"source": "elpais", "url": "http://a", "date": "2026-02-01"}

    merge_article(wv, "obj-1", article)

    _, payload = wv.update.call_args[0]
    assert payload["sources"] == ["elpais"]
    assert payload["urls"] == ["http://a"]
    assert payload["date"] == "2026-01-01"   # existing is older, kept
