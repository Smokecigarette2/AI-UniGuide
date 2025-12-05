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

]

# ---------- ЛОКАЛЬНАЯ МИНИ-БАЗА ПО KIMEP ----------

KIMEP_FACTS = [
    "KIMEP University — частный университет в Алматы с фокусом на бизнесе, экономике, праве, государственном управлении и международных отношениях.",
    "Кампус KIMEP расположен в центре Алматы по адресу: проспект Абая, 4 (угол улицы Валиханова).",
    "Приёмная комиссия (Admissions Office) для абитуриентов находится в Валихановском корпусе KIMEP, на первом этаже.",
    "Офис регистратора (Registrar’s Office), который отвечает за регистрацию на курсы и академические справки, расположен в Валихановском корпусе на втором этаже.",
    "Общежитие KIMEP (Residence Hall) находится рядом с основным кампусом и предназначено в первую очередь для студентов из регионов и иностранных студентов.",
    "Заселение в общежитие происходит через администрацию Residence Hall: студенты подают заявку, получают подтверждение, оформляют документы и заселяются в комнату.",
    "Медицинский центр KIMEP находится внутри общежития и оказывает базовую медицинскую помощь студентам и сотрудникам.",
    "Главная библиотека KIMEP — Olivier Giscard d’Estaing Library — находится на кампусе и предлагает зоны тихой работы, групповые комнаты и доступ к электронным ресурсам.",
    "KIMEP использует кредитную систему обучения по североамериканской модели: студенты сами формируют индивидуальный учебный план из обязательных и элективных курсов.",
    "Многие программы KIMEP доступны полностью на английском языке, часть — на казахском и русском; для англоязычных программ часто требуется подтверждение уровня английского.",
    "У студентов KIMEP есть онлайн-личный кабинет, где они регистрируются на курсы, смотрят расписание, оценки и финансовую информацию.",
    "Карьерный центр KIMEP помогает студентам со стажировками и поиском работы, проводит ярмарки вакансий и тренинги по резюме и собеседованиям."
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
    # 1) Собираем промпт по universityId
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
    elif req.universityId == "kimep":
        context = "\n".join(f"• {fact}" for fact in KIMEP_FACTS)
        instructions = (
            "Ты — AI-гид по KIMEP University в Алматы. "
            "Отвечай по-русски, опираясь только на факты из базы ниже. "
            "Если точного ответа нет, честно скажи, что такой информации нет в базе "
            "и посоветуй обратиться в приёмную или приёмный офис KIMEP.\n\n"
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
