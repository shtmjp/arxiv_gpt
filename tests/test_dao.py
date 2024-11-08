from db import GSSPaperDAO, arxivresult2paperinfo
from log import get_logger_for_test
from main import search_arxiv, summarize_paper


def test_gsspaperdao() -> None:
    logger = get_logger_for_test(__name__)
    query = "ti:%22 Frequentist Oracle Properties of Bayesian Stacking Estimators %22"
    results = search_arxiv(query, max_results=1)
    result = results[0]
    summary = summarize_paper(result)
    logger.debug(summary)

    info = arxivresult2paperinfo(result, summary)
    dao = GSSPaperDAO()
    dao.add_paper(info)
    logger.info(dao.get_paper_ids("arxiv"))
