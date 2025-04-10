from typing import List, Dict
from huggingface_hub import InferenceClient
from database.database import get_book_full_text, save_users_db, get_total_pages

# Инициализация клиента для модели анализа (например, DeepSeek)
client = InferenceClient(provider="hf-inference", api_key="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")
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# ------------------------------
# Функция для определения типа текста
def determine_text_type(text: str) -> str:
    """
    Определить тип текста (например, учебный, научно-популярный, научный, художественный)
    на основе его фрагмента. Возвращается один из типов.
    """
    PROMPT_TEMPLATE = '''Analyze the given text and determine its type (e.g., scientific, literary, journalistic, technical, philosophical, etc.). Based on the identified type, extract only the key parameters that characterize this type of text.  

        Output the result strictly as a comma-separated list of parameters in russian without any additional text.  

        Examples of key parameters for different text types:  
        - **Literary: персонажи, сюжет, настроение, стиль повествования, конфликты, символика, описание среды  
        - **Scientific: основные термины, гипотезы, методы, доказательства, выводы  
        - **Journalistic: ключевые события, участники, место, время, аргументы  
        - **Technical: предмет описания, термины, инструкции, алгоритмы, примеры  
        - **Philosophical: основные идеи, аргументы, философские термины, парадоксы, концепции  

        Text:  
        "{text}"  
        
        '''
    # Берём средний фрагмент текста
    mid_point = len(text) // 2
    fragment = text[max(mid_point - 500, 0): mid_point + 500]

    messages = [
        {
            "role": "user",
            "content": (
                PROMPT_TEMPLATE.format(text=fragment)
            )
        }
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100
    )
    text_type = completion.choices[0].message.content.strip()
    return text_type