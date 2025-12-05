# backend/main.py

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

# ---------- НАСТРОЙКА OPENAI ЧЕРЕЗ HTTP ----------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"        # можно любой актуальный текстовый
OPENAI_URL = "https://api.openai.com/v1/responses"

# ---------- ЛОКАЛЬНАЯ БАЗА ПО NARXOZ ----------

NARXOZ_FACTS = [
    "Приёмная комиссия Narxoz University в основной период работы находится в главном корпусе, кабинет 102.",
    "Консультации по вопросам поступления на бакалавриат проходят в кабинете 104 с 9:00 до 18:00 по будням.",
    "Академический офис бакалавриата расположен в кабинете 215 главного корпуса.",
    "Академический офис магистратуры и PhD находится в кабинете 312.",
    "Касса университета и отдел оплаты обучения расположены на первом этаже, кабинет 110.",
    "Центр карьерного развития находится в кабинете 210 и помогает студентам с поиском стажировок.",
    "Учебные аудитории с номерами 100–199 находятся на первом этаже главного корпуса.",
    "Аудитории 200–299 находятся на втором этаже, в том числе большие лекционные залы.",
    "Аудитории 300–399 расположены на третьем этаже и чаще всего используются для семинаров.",
    "Компьютерные лаборатории по цифровым технологиям находятся в аудиториях 220, 221 и 222.",
    "Библиотека Narxoz University расположена в отдельном крыле второго этажа.",
    "Читальный зал библиотеки работает с 9:00 до 21:00 по будням и с 10:00 до 18:00 по субботам.",
    "В библиотеке есть тихая зона для самостоятельной работы и открытые пространства для групповых проектов.",
    "Студенческий коворкинг с розетками и Wi-Fi находится на первом этаже рядом с кафетерием.",
    "Студенческое общежитие Narxoz располагается в пешей доступности от кампуса и имеет несколько блоков.",
    "Заселение в общежитие проводится через отдел по работе со студентами, кабинет 205.",
    "Для получения места в общежитии приоритет отдается студентам из регионов и льготным категориям.",
    "Учебный год делится на осенний и весенний семестры, каждый примерно по 15 учебных недель.",
    "Экзаменационная сессия проходит в конце каждого семестра и длится 2–3 недели.",
    "Для связи с деканатом студенты могут использовать корпоративную почту и личные обращения в приёмные часы."
]

# ---------- FASTAPI ----------

app = FastAPI(title="AI-UniGuide backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    universityId: str | None = None


class AskResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def ask_ai(req: AskRequest):
    # 1) Собираем промпт
    if req.universityId == "narxoz":
        context = "\n".join(f"• {fact}" for fact in NARXOZ_FACTS)
        instructions = (
            "Ты — AI-гид по Narxoz University в Алматы. "
            "Отвечай по-русски, опираясь только на факты из базы ниже. "
            "Если точного ответа нет, честно скажи, что такой информации нет в базе "
            "и посоветуй обратиться в приёмную комиссию.\n\n"
            f"БАЗА ФАКТОВ:\n{context}\n\n"
            f"Вопрос пользователя: {req.question}"
        )
    else:
        instructions = (
            "Ты — AI-помощник по выбору университетов Казахстана. "
            "Отвечай кратко и по делу, максимум 4–5 предложений.\n\n"
            f"Вопрос пользователя: {req.question}"
        )

    # 2) Готовим запрос к Responses API
    payload = {
        "model": OPENAI_MODEL,
        "input": instructions,
        "max_output_tokens": 400,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # 3) Делаем запрос к OpenAI
    try:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY не найден в переменных окружения")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                OPENAI_URL,
                headers=headers,
                json=payload,
            )

        resp.raise_for_status()
        data = resp.json()

        # 4) Аккуратно достаём текст из output
        answer_parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        answer_parts.append(c.get("text", ""))

        if not answer_parts:
            # fallback на самый первый кусок, если структура вдруг другая
            answer = (
                data.get("output", [{}])[0]
                    .get("content", [{}])[0]
                    .get("text", "")
                    .strip()
            )
        else:
            answer = "".join(answer_parts).strip()

    except Exception as e:
        print("OpenAI error:", repr(e))
        answer = "Не получилось получить ответ от AI."

    return AskResponse(answer=answer)
