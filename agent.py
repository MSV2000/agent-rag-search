import json
import os
import re
from dotenv import load_dotenv
from searcher import google_search, collect_for_llm

# Загружаем переменные из .env файла
load_dotenv()

# Получаем переменные
search_id = os.getenv('SEARCH_ID')
api_key = os.getenv('API_KEY')


class Agent:
    """
    Агент для взаимодействия с LLM-моделью с поддержкой вызова внешних инструментов

    Класс реализует агентный подход:
    - формирует системные промпты;
    - запрашивает LLM;
    - определяет необходимость вызова инструмента;
    - выполняет поиск через вызов web_search;
    - повторно обращается к LLM с расширенным контекстом.
    """

    def __init__(self, llm):
        """
        Инициализация класса

        Args:
            llm: LLM-модель
        """
        self.llm = llm
        self.default_system_prompt = (
            "Ты — интеллектуальный помощник, задача которого — отвечать на вопросы пользователя строго на основе "
            "предоставленного контекста и истории диалога. Ты не должен придумывать информацию, которой нет в "
            "контексте или истории диалога. Если ответа на вопрос нет в контексте, скажи об этом прямо. "
            "Ты не должен писать рассказы, стихи, сочинять истории или "
            "создавать любой другой творческий контент. Твои ответы должны быть строго основаны на фактах, данных "
            "и предоставленном контексте."
        )
        self.agent_system_prompt = (
            "Ты — интеллектуальный помощник, задача которого — отвечать на вопросы пользователя строго на основе "
            "предоставленного контекста и истории диалога. "
            "Если информации из контекста недостаточно для ответа, "
            "ты ДОЛЖЕН использовать инструмент.\n\n"

            "Доступные инструменты:\n\n"

            "Инструмент: web_search\n"
            "Описание: поиск актуальной информации в интернете\n"
            "Параметры:\n"
            "- query (string): поисковый запрос\n"
            "- reason (string): зачем нужен поиск\n\n"

            "Формат вызова инструмента (СТРОГО JSON, без текста):\n\n"

            "{\n"
            '  "name": "web_search",\n'
            '  "arguments": {\n'
            '    "query": "...",\n'
            '    "reason": "..."\n'
            "  }\n"
            "}\n\n"

            "Ты не должен писать рассказы, стихи, сочинять истории или создавать любой другой творческий контент. "
            "Твои ответы должны быть строго основаны на фактах, данных и предоставленном контексте."
        )

    @staticmethod
    def _parse_tool_call(text: str) -> dict | None:
        """
        Извлекает и валидирует вызов инструмента из ответа LLM-модели

        Ожидает, что вызов инструмента представлен в виде JSON-объекта
        с именем `web_search` и аргументами `query` и `reason`

        Args:
            text: Текст ответа LLM-модели

        Returns:
            dict | None: Словарь аргументов инструмента или None, если корректный вызов не найден
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None

        json_text = match.group(0)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError:
            return None

        if data.get("name") != "web_search":
            return None

        args = data.get("arguments", {})
        if "query" not in args or "reason" not in args:
            return None

        return args

    def _call_llm(self, messages: list[dict], tool_call_mode: bool = False) -> str:
        """
        Формирует промт и выполняет запрос к LLM-модели

        Args:
            messages: Список сообщений
            tool_call_mode: Флаг вызова инструмента. Если True — модель ожидается к вызову инструмента

        Returns:
            str: Ответ LLM-модели
        """
        prompt = self.llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return self.llm.generate(prompt, tool_call_mode)

    def run(self, query: str, context: str = "") -> str:
        """
        Запускает агентный цикл обработки запроса

        1. Передаёт запрос и контекст в LLM-модель с агентным промптом
        2. Проверяет, требуется ли вызов инструмента
        3. При необходимости выполняет web-поиск и расширяет контекст
        4. Повторно обращается к LLM для получения финального ответа

        Args:
            query: Вопрос пользователя
            context: Контекст, на основе которого нужно отвечать

        Returns:
            str: Итоговый ответ LLM-модели
        """
        messages = [
            {"role": "system", "content": self.agent_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Контекст:\n{context}\n\n"
                    f"Вопрос:\n{query}"
                )
            }
        ]

        first_response = self._call_llm(messages, tool_call_mode=True)

        tool_args = self._parse_tool_call(first_response)

        if not tool_args:
            return first_response

        search_result = google_search(tool_args["query"], search_id, api_key)
        search_result = collect_for_llm(search_result)

        context += f"\n===========\nИнформация из интернет источников\n{search_result}\n===========\n"
        messages = [
            {"role": "system", "content": self.default_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Контекст:\n{context}\n\n"
                    f"Вопрос:\n{query}"
                )
            }
        ]

        final_response = self._call_llm(messages, tool_call_mode=False)
        return final_response
