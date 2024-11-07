import google.generativeai as genai
import settings
from log import get_logger_for_test

genai.configure(api_key=settings.GEMINI_API_KEY)


def test_gemimi() -> None:
    logger = get_logger_for_test(__name__)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("The opposite of hot is")
    assert isinstance(response.text, str)
    logger.debug(response.text)
