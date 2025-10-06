import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass

import arxiv
import google.generativeai as genai
from discord_webhook import DiscordWebhook

from db import AbstractPaperDAO, GSSPaperDAO, arxivresult2paperinfo
from prompts import SUMMARY_PREFIX
from settings import (
    EXTRA_ARXIV_SEARCH_CONFIGS,
    GEMINI_API_KEY,
)

genai.configure(api_key=GEMINI_API_KEY)


def summarize_paper(result: arxiv.Result) -> str:
    # get prompt
    prompt = SUMMARY_PREFIX
    prompt += f"title: {result.title}\nbody: {result.summary}"
    # generate summary
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    summary = response.text
    title_en = result.title
    title, *body = summary.split("\n")
    body = "\n".join(body)
    date_str = result.published.strftime("%Y-%m-%d %H:%M:%S")

    return f"発行日: {date_str}\n{result.entry_id}\n{title_en}\n{title}\n{body}\n"


@dataclass(frozen=True)
class SearchTask:
    query: str
    webhook_url: str


def load_search_tasks() -> list[SearchTask]:
    """Load search tasks from settings."""
    # tasks = [SearchTask(query=SEARCH_QUERY, webhook_url=DISCORD_WEBHOOK_URL)]
    tasks = []

    if not EXTRA_ARXIV_SEARCH_CONFIGS:
        return tasks

    try:
        # EXTRA_ARXIV_SEARCH_CONFIGS is path to a JSON file
        with open(EXTRA_ARXIV_SEARCH_CONFIGS, encoding="utf-8") as f:
            raw_configs = json.loads(f.read())
    except json.JSONDecodeError as exc:  # pragma: no cover - configuration error
        msg = "Failed to parse EXTRA_ARXIV_SEARCH_CONFIGS as JSON."
        raise ValueError(msg) from exc

    if not isinstance(raw_configs, list):  # pragma: no cover - configuration error
        msg = "EXTRA_ARXIV_SEARCH_CONFIGS must be a JSON array."
        raise ValueError(msg)

    for raw in raw_configs:
        if not isinstance(raw, dict):  # pragma: no cover - configuration error
            msg = "Each extra search config must be a JSON object."
            raise ValueError(msg)

        query = raw.get("query")
        if (
            not isinstance(query, str) or not query
        ):  # pragma: no cover - configuration error
            msg = "Each extra search config must include a non-empty 'query'."
            raise ValueError(msg)

        webhook_url = raw.get("webhook_url")
        webhook_env = raw.get("webhook_env")

        if (webhook_url is None) == (webhook_env is None):
            msg = (
                "Each extra search config must include either 'webhook_url' or "
                "'webhook_env', but not both."
            )
            raise ValueError(msg)

        if webhook_env is not None:
            if not isinstance(webhook_env, str) or not webhook_env:
                msg = "'webhook_env' must be a non-empty string."
                raise ValueError(msg)
            try:
                webhook_url = os.environ[webhook_env]
            except KeyError as exc:  # pragma: no cover - configuration error
                msg = f"Environment variable '{webhook_env}' is not set."
                raise KeyError(msg) from exc
        elif not isinstance(webhook_url, str) or not webhook_url:
            msg = "'webhook_url' must be a non-empty string when provided."
            raise ValueError(msg)

        tasks.append(SearchTask(query=query, webhook_url=webhook_url))

    return tasks


def search_arxiv(query: str, max_results: int) -> list[arxiv.Result]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,  # 検索クエリ
        max_results=max_results,  # 取得する論文数
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 論文を投稿された日付でソートする
        sort_order=arxiv.SortOrder.Descending,  # 新しい論文から順に取得する
    )
    return list(client.results(search))


def post_summaries(
    dao: AbstractPaperDAO,
    tasks: Iterable[SearchTask],
    existing_ids: set[str],
) -> None:
    for task in tasks:
        # arxiv APIで最新の論文情報を取得する
        result_list = search_arxiv(task.query, max_results=100)

        # すでにDBにある論文は除外する
        result_list = [r for r in result_list if r.entry_id not in existing_ids]

        # ランダムにnum_papersの数だけ選ぶ
        num_papers = min(3, len(result_list))
        if num_papers == 0:
            continue

        results = random.sample(result_list, k=num_papers)

        for result in results:
            # 要約をdiscordに投稿
            summary = summarize_paper(result)
            webhook = DiscordWebhook(url=task.webhook_url, content=summary)
            webhook.execute()

            # databaseに追加
            info = arxivresult2paperinfo(result, summary)
            dao.add_paper(info)
            existing_ids.add(result.entry_id)

            # 5秒待つ
            time.sleep(5)


def main() -> None:
    tasks = load_search_tasks()

    dao: AbstractPaperDAO = GSSPaperDAO()
    existing_ids = set(dao.get_paper_ids("arxiv"))

    post_summaries(dao=dao, tasks=tasks, existing_ids=existing_ids)


if __name__ == "__main__":
    main()
