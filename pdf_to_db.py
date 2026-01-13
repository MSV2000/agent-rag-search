import chromadb
import fitz
import os
import shutil
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


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

        self.path_db = path_db

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
        # Проверка корректности start_page
        if start_page < 1:
            raise ValueError("Номер страницы должен быть положительным числом")

        # Проверка корректности пути
        if not isinstance(file_path, str) or not file_path:
            raise ValueError("Некорректный путь к файлу")

        # Проверка существования файла
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

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
                      start_page: int = 1, overwrite: bool = False) -> None:
        """
        Извлечение текста из PDF файла и добавление в векторную базу данных

        Args:
            file_path: путь к PDF файлу
            collection_name: название коллекции
            text_splitter: объект для разделения текста
            start_page: номер страницы с которой начинать извлечение (по умолчанию 1)
            overwrite: перезапись существующей коллекции или добавление к ней

        Raises:
            ValueError: если файл или collection_name пустые
        """
        # Проверка корректности collection_name
        if collection_name == "":
            raise ValueError(f"Название коллекции не должно быть пустым")

        # Извлечение текста из PDF файла
        text = self.extract_text_from_pdf(file_path=file_path, start_page=start_page)

        # Проверка, что PDF не пустой
        if not text or text.strip() == "":
            raise ValueError(f"PDF файл {file_path} не содержит текст")

        if text_splitter:
            split_text = text_splitter.split_text(text=text)
        else:
            split_text = [text]

        # Добавляем в базу данных
        self.add_texts_to_db(
            text=split_text,
            collection_name=collection_name,
            overwrite=overwrite
        )

    def _collection_path(self, collection_name: str) -> str:
        """
        Формирование пути к директории конкретной коллекции

        Args:
            collection_name: название коллекции

        Returns:
            str: полный путь к директории коллекции
        """
        return os.path.join(self.path_db, collection_name)

    def add_texts_to_db(self, text: str | list[str], collection_name: str, overwrite: bool = False) -> None:
        """
        Сохранение текста в векторную базу данных с разделением по коллекциям

        Args:
            text: список строк текста
            collection_name: название коллекции
            overwrite: перезапись существующей коллекции или добавление к ней
        """
        # Преобразование текста к списку, если нужно
        if not isinstance(text, list):
            text = [text]

        collection_path = self._collection_path(collection_name)

        if overwrite and os.path.exists(collection_path):
            shutil.rmtree(collection_path)

        if overwrite:
            # Создание новой коллекции или перезаписывание существующей
            Chroma.from_texts(
                texts=text,
                persist_directory=collection_path,
                embedding=self.embedding_function,
            )
        else:
            # Добавление в существующую коллекцию
            db = Chroma(
                persist_directory=collection_path,
                embedding_function=self.embedding_function,
            )
            db.add_texts(text)

    def load_collection(self, collection_name: str) -> VectorStoreRetriever:
        """
        Загрузка существующей коллекции

        Args:
            collection_name: название коллекции

        Returns:
            retriever: объект для поиска по векторной базе

        Raises:
            ValueError: если collection_name пустой
        """
        # Проверка корректности collection_name
        if collection_name == "":
            raise ValueError(f"Название коллекции не должно быть пустым")

        db = Chroma(
            persist_directory=self._collection_path(collection_name),
            embedding_function=self.embedding_function,
        )

        # Создание объекта для поиска по векторной базе (топ 10 похожих результатов)
        db_retriever = db.as_retriever(search_kwargs={"k": 10})
        return db_retriever

    def list_collection(self) -> list[str]:
        """
        Получение списка существующих коллекций

        Returns:
            list[str]: список с названиями коллекций
        """
        if not os.path.exists(self.path_db):
            return []

        return [name for name in os.listdir(self.path_db) if os.path.isdir(os.path.join(self.path_db, name))]

    def delete_collection(self, collection_name: str) -> None:
        """
        Удаление существующей коллекции

        Args:
            collection_name: название коллекции

        Raises:
            ValueError: если collection_name пустой или коллекции не существует
        """
        if not collection_name:
            raise ValueError("Название коллекции не должно быть пустым")

        collection_path = self._collection_path(collection_name)

        if not os.path.exists(collection_path):
            raise ValueError(f"Коллекция '{collection_name}' не существует")

        shutil.rmtree(collection_path)


if __name__ == "__main__":
    pdf_db = PDFVecDataBase(embeddings_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                            path_db="chroma_db")

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500,
                                              separators=["\n\n", "\n", ",", " ", ""])

    pdf_db.add_pdf_to_db(file_path="example.pdf", collection_name="collection_1", text_splitter=splitter,
                         start_page=2, overwrite=True)

    # pdf_db.add_texts_to_db(text="text", collection_name="collection_1", overwrite=True)

    # retriever = pdf_db.load_collection(collection_name="collection_1")

    pdf_db.delete_collection(collection_name="collection_1")
