import arxiv

from log import get_logger_for_test


def test_arxiv_search() -> None:
    logger = get_logger_for_test(__name__)
    query = "ti:%22 point process %22"

    # arxiv APIで最新の論文情報を取得する
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,  # 検索クエリ
        max_results=100,  # 取得する論文数
        sort_by=arxiv.SortCriterion.SubmittedDate,  # 論文を投稿された日付でソートする
        sort_order=arxiv.SortOrder.Descending,  # 新しい論文から順に取得する
    )
    results = client.results(search)
    for i, result in enumerate(results):
        logger.debug(result.title)
        logger.debug(result.published)
        th = 3
        if i == th:
            break
