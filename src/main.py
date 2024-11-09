import random
import time

import arxiv
import google.generativeai as genai
from discord_webhook import DiscordWebhook

from db import AbstractPaperDAO, GSSPaperDAO, arxivresult2paperinfo
from prompts import SEARCH_QUERY, SUMMARY_PREFIX
from settings import DISCORD_WEBHOOK_URL, GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)


def summarize_paper(result: arxiv.Result) -> str:
    # get prompt
    prompt = SUMMARY_PREFIX
    prompt += f"title: {result.title}\nbody: {result.summary}"
    # generate summary
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    summary = response.text
    title_en = result.title
    title, *body = summary.split("\n")
    body = "\n".join(body)
    date_str = result.published.strftime("%Y-%m-%d %H:%M:%S")

    return f"発行日: {date_str}\n{result.entry_id}\n{title_en}\n{title}\n{body}\n"


def search_arxiv(query: str, max_results: int) -> list[arxiv.Result]:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,  # 検索クエリ
        max_results=max_results,  # 取得する論文数
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 論文を投稿された日付でソートする
        sort_order=arxiv.SortOrder.Descending,  # 新しい論文から順に取得する
    )
    return list(client.results(search))


def main() -> None:
    # arxiv APIで最新の論文情報を取得する
    result_list = search_arxiv(SEARCH_QUERY, max_results=100)

    # すでにDBにある論文は除外する
    # in set の計算量はO(1)なので、idsをsetに変換しておく
    dao: AbstractPaperDAO = GSSPaperDAO()
    ids = dao.get_paper_ids("arxiv")
    result_list = [r for r in result_list if r.entry_id not in set(ids)]
    # ランダムにnum_papersの数だけ選ぶ
    num_papers = min(3, len(result_list))
    results = random.sample(result_list, k=num_papers)  # random.sample(l, 0) = []

    summary_list: list[str] = [""] * num_papers
    for i, result in enumerate(results):
        # 要約をdiscordに投稿
        summary = summarize_paper(result)
        webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL, content=summary)
        webhook.execute()
        summary_list[i] = summary

        # databaseに追加
        info = arxivresult2paperinfo(result, summary)
        dao.add_paper(info)

        # 5秒待つ
        time.sleep(5)


if __name__ == "__main__":
    main()
