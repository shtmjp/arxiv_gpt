import random

import arxiv
import google.generativeai as genai
from discord_webhook import DiscordWebhook

from prompts import SEARCH_QUERY, SUMMARY_PREFIX
from settings import DISCORD_WEBHOOK_URL, GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)


def get_summary(result: arxiv.Result) -> str:
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


def main() -> None:
    query = SEARCH_QUERY

    # arxiv APIで最新の論文情報を取得する
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,  # 検索クエリ
        max_results=200,  # 取得する論文数
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 論文を投稿された日付でソートする
        sort_order=arxiv.SortOrder.Descending,  # 新しい論文から順に取得する
    )
    # searchの結果をリストに格納
    result_list = list(client.results(search))
    # ランダムにnum_papersの数だけ選ぶ
    num_papers = 3
    results = random.sample(result_list, k=num_papers)

    # 論文情報をSlackに投稿する
    for result in results:
        summary = get_summary(result)
        webhook = DiscordWebhook(url=DISCORD_WEBHOOK_URL, content=summary)
        webhook.execute()


if __name__ == "__main__":
    main()
