from typing import List, Dict
from huggingface_hub import InferenceClient
from database.database import get_book_full_text, save_users_db, get_total_pages
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import os

# Инициализация клиента модели
client = InferenceClient(provider="hf-inference", api_key="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")
model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Подсчёт токенов
def count_tokens(text, model_name="cl100k_base"):
    enc = tiktoken.get_encoding(model_name)
    return len(enc.encode(text))

# Создание/загрузка векторной базы
embedding_model = HuggingFaceEmbeddings(model_name="ai-forever/ru-en-RoSBERTa")
vectorstore_path = "faiss_index"

if os.path.exists(vectorstore_path):
    db = FAISS.load_local(vectorstore_path, embedding_model)
else:
    db = None

# Функция построения индекса для книги
def build_index_from_book(user_id: str, book_name: str):
    full_text = get_book_full_text(book_name, user_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    global db
    db = FAISS.from_documents(docs, embedding_model)
    db.save_local(vectorstore_path)

# Поиск релевантного контекста
def retrieve_context_dynamic(query, k=30, max_tokens=3000):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    relevant_docs = retriever.get_relevant_documents(query)

    context = ""
    total_tokens = 0

    for doc in relevant_docs:
        chunk = doc.page_content
        chunk_tokens = count_tokens(chunk)
        if total_tokens + chunk_tokens > max_tokens:
            break
        context += chunk + "\n---\n"
        total_tokens += chunk_tokens

    return context

# Генерация ответа

def ask_question(question: str, chat_history: list, answer_mode: str = "detailed") -> str:
    context = retrieve_context_dynamic(question)
    messages = [
        {"role": "system", "content":
            "You are an AI assistant specializing in analyzing the content of books. "
            "Answer questions in detail, analyzing the style, symbolism, and themes of the work. Answer only in Russian."
        },
    ] + chat_history + [
        {"role": "user", "content": f"Вот текущий контекст книги:\n{context}\nВопрос: {question}"}
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=700
    )
    detailed_answer = completion.choices[0].message.content

    if answer_mode != "detailed":
        if answer_mode == "short":
            target_size = 50
        elif answer_mode == "medium":
            target_size = 100
        else:
            target_size = 200
        from ai_tools.summarize_system import compress_text
        adjusted_answer = compress_text(detailed_answer, final_target_size=target_size)
        final_answer = adjusted_answer
    else:
        final_answer = detailed_answer

    return final_answer
