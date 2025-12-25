import os
import fitz
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


class PDFVecDataBase:
    """Класс для парсинга и добавления pdf в базу данных"""

    def __init__(self, embeddings_model: str, path_db: str) -> None:
        """
        Инициализация класса

        Args:
            embeddings_model: имя модели эмбеддингов
            path_db: путь до векторной базы данных
        """

        # Инициализация модели эмбеддингов (будет на CPU)
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={"device": "cpu"}
        )

    @staticmethod
    def extract_text_from_pdf(file_path: str, start_page: int = 1) -> str:
        """
        Извлечение текста из PDF файла

        Args:
            file_path: путь к PDF файлу
            start_page: номер страницы с которой начинать извлечение (по умолчанию 1)

        Returns:
            str: извлеченный текст

        Raises:
            FileNotFoundError: если файл не существует
            ValueError: если file_path или start_page некорректен
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        if not isinstance(file_path, str) or not file_path:
            raise ValueError("Некорректный путь к файлу")

        if start_page < 1:
            raise ValueError("Номер страницы должен быть положительным числом")

        doc = None
        try:
            doc = fitz.open(file_path)

            # Проверка корректности start_page
            if start_page > len(doc):
                raise ValueError(f"Стартовая страница {start_page} превышает количество страниц {len(doc)}")

            # Сбор текста
            text_parts = []
            for page_num in range(start_page, len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text:  # Добавляем только непустой текст
                    text_parts.append(page_text)

            return "".join(text_parts)

        except fitz.FileDataError as e:
            raise ValueError(f"Ошибка чтения PDF файла: {str(e)}")
        finally:
            if doc:
                doc.close()

    def add_pdf_to_db(self, file_path: str, collection_name: str, text_splitter: RecursiveCharacterTextSplitter = None,
                      start_page: int = 1):
        """
        Извлечение текста из PDF файла и добавление в векторную базу данных

        Args:
            file_path: путь к PDF файлу
            collection_name: название коллекции
            text_splitter: объект для разделения текста
            start_page: номер страницы с которой начинать извлечение (по умолчанию 1)

        Returns:
            retriever: объект для поиска по векторной базе

        Raises:
            ValueError: если файл или collection_name пустые
        """
        if collection_name == "":
            raise ValueError(f"Название коллекции не должно быть пустым")

        text = self.extract_text_from_pdf(file_path=file_path, start_page=start_page)

        if not text or text.strip() == "":
            raise ValueError(f"PDF файл {file_path} не содержит текст")

        if text_splitter:
            split_text = text_splitter.split_text(text=text)
        else:
            split_text = [text]

        # Добавляем в базу данных
        return self.add_texts_to_db(
            text=split_text,
            collection_name=collection_name,
            overwrite=False
        )

    def add_texts_to_db(self, text: List[str], collection_name: str, overwrite: bool = False):
        pass


if __name__ == "__main__":
    pdf_db = PDFVecDataBase(embeddings_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                            path_db="chroma_db")

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500,
                                              separators=["\n\n", "\n", ",", " ", ""])

    pdf_db.add_pdf_to_db(file_path="example.pdf", collection_name="", text_splitter=splitter, start_page=2)
