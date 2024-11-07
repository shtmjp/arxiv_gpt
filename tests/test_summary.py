import arxiv

from log import get_logger_for_test
from main import get_summary


def test_get_summary() -> None:
    logger = get_logger_for_test(__name__)
    query = "bayesian inference for logistic models \
    using Polya-Gamma latent variables"
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,  # 検索クエリ
        max_results=3,  # 取得する論文数
    )
    result = next(client.results(search))
    s = get_summary(result)
    assert isinstance(s, str)
    logger.debug(s)
