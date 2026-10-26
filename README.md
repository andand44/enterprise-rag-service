# Enterprise RAG Service

Промышленный микросервис на базе FastAPI для создания умных корпоративных ассистентов. Проект реализует передовые практики разработки LLM-приложений и готов к развертыванию в продакшене.

## 🏗 Ключевые возможности

- **Асинхронный стриминг (SSE)**: Ответ от модели передается клиенту в реальном времени (посимвольно), как в ChatGPT, благодаря Server-Sent Events и `ollama.AsyncClient`.
- **Продвинутый RAG (Reranking)**: Используется двухступенчатый поиск. Сначала Qdrant находит 5 релевантных документов по векторам, затем модель `Flashrank` (Cross-Encoder) переранжирует их и оставляет только 2 самых точных.
- **Наблюдаемость (Observability)**: Полная интеграция с **Langfuse** (v4, OpenTelemetry). Каждый запрос логируется в виде дерева трейсов, что позволяет отслеживать время выполнения RAG-поиска, генерации и расход токенов.
- **CI/CD (GitHub Actions)**: При каждом пуше автоматически запускаются тесты (`pytest`) с использованием моков (mocking) для изоляции от внешних сервисов.

## 🧰 Стек технологий

- **Python 3.12+**
- **FastAPI** (Асинхронный API)
- **LangChain & LangChain-Qdrant** (Оркестрация RAG)
- **Ollama** (Локальная LLM `qwen3.5` и эмбеддинги `nomic-embed-text`)
- **Qdrant** (Векторная база данных, запускается в Docker)
- **Flashrank** (Reranking)
- **Langfuse** (Observability)
- **Pytest** (Автотесты)
- **Docker & Docker Compose**

## 🚀 Запуск проекта

### 1. Подготовка окружения
Убедитесь, что у вас запущена Ollama и скачаны модели:
```bash
ollama pull qwen3.5:2b
ollama pull nomic-embed-text
```

## 2. Настройка переменных окружения

Создайте файл .env в корне проекта и добавьте ваши ключи от Langfuse:
```bash
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

## 3. Запуск инфраструктуры и API

Запустите векторную базу данных Qdrant:
```bash
docker compose up -d
```

Установите зависимости и запустите FastAPI:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

API и Swagger-документация будут доступны по адресу: `bashhttp://127.0.0.1:8000/docs`

## 🧪 Тестирование

Проект покрыт автотестами (с использованием моков для Qdrant и Ollama).
```bash
pytest -v
```

## 📡 Примеры эндпоинтов

`POST /add_knowledge` — добавить факт в базу знаний.

`POST /chat/stream` — задать вопрос агенту и получить потоковый ответ (SSE).