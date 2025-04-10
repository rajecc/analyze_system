from typing import List, Dict
from huggingface_hub import InferenceClient
from database.database import get_book_full_text, save_users_db, get_total_pages

# Инициализация клиента для модели анализа (например, DeepSeek)
client = InferenceClient(provider="hf-inference", api_key="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")
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Функция для извлечения тегов и жанров из текста
def extract_tags_and_genres(text: str) -> Dict[str, List[str]]:
    """
    Извлечь теги и жанры из переданного текста.
    Теги извлекаются посредством запроса к модели, а жанры определяются эвристически.
    """
    # Извлечение тегов
    response_tags = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                    "Extract tags from the following text about books preferences. "
                    "Write only tags separated by commas. The first letter of each tag must be capitalized.: " + text
            )
        }],
        max_tokens=500
    )
    tags = [tag.strip() for tag in response_tags.choices[0].message.content.split(",") if tag.strip()]

    # Эвристическое определение жанров на основе ключевых слов
    possible_genres = {
        "учебный": ["образование", "учебный"],
        "научно-популярный": ["наука", "популярный"],
        "научный": ["исследование", "теория"],
        "художественный": ["роман", "рассказ"]
    }
    genres = []
    text_lower = text.lower()
    for genre, keywords in possible_genres.items():
        if any(keyword in text_lower for keyword in keywords):
            genres.append(genre)
    if not genres:
        genres.append("художественный")

    return {"tags": tags, "genres": genres}