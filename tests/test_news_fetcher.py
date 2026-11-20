import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import requests

from news_fetcher import NewsFetcher


class NewsFetcherTests(unittest.TestCase):
    def test_samples_have_an_explicit_source_and_no_publisher_link(self):
        with patch.dict("os.environ", {"NEWS_API_KEY": ""}):
            rows = NewsFetcher().get_news("ABC")
        self.assertTrue(rows)
        self.assertTrue(all(row["is_sample"] and row["source"]["name"] == "Synthetic example" and not row["url"] for row in rows))

    def test_live_rows_keep_the_provider_source(self):
        row = {"source": {"name": "Example publisher"}, "title": "A real response"}
        reply = Mock(status_code=200)
        reply.json.return_value = {"articles": [row]}
        with patch.dict("os.environ", {"NEWS_API_KEY": "fixture"}), patch("news_fetcher.requests.get", return_value=reply) as get:
            self.assertEqual(NewsFetcher().get_news("ABC"), [row])
        self.assertNotIn("apiKey", get.call_args.kwargs["params"])
        self.assertEqual(get.call_args.kwargs["headers"]["X-Api-Key"], "fixture")

    def test_offline_retry_uses_labeled_examples_and_limits_log_content(self):
        out = io.StringIO()
        with patch.dict("os.environ", {"NEWS_API_KEY": "fixture"}), redirect_stdout(out):
            with patch("news_fetcher.requests.get", side_effect=requests.Timeout("sensitive request metadata")):
                rows = NewsFetcher().get_news("ABC")
        self.assertTrue(all(row["is_sample"] for row in rows))
        self.assertNotIn("sensitive request metadata", out.getvalue())
