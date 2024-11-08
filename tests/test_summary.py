from log import get_logger_for_test
from main import search_arxiv, summarize_paper


def test_get_summary() -> None:
    logger = get_logger_for_test(__name__)
    query = "bayesian inference for logistic models \
    using Polya-Gamma latent variables"
    results = search_arxiv(query, max_results=1)
    summary = summarize_paper(results[0])
    logger.debug(summary)
