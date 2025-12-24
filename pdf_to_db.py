import os
import fitz
from langchain_huggingface import HuggingFaceEmbeddings


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

    def add_pdf_to_db(self, file_path: str, text_splitter, start_page: int = 2):
        pass


if __name__ == "__main__":
    pdf_db = PDFVecDataBase(embeddings_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                            path_db="chroma_db")

    print(pdf_db.extract_text_from_pdf(file_path="example.pdf", start_page=2))
