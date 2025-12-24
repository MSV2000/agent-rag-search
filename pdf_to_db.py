import os
import fitz


def extract_text_from_pdf(file_path: str, start_page: int = 1) -> str:
    """
    Извлечение текста из PDF файла

    Args:
        file_path: путь к PDF файлу
        start_page: номер страницы с которой начинать извлечение (по умолчанию 2)

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


if __name__ == "__main__":
    print(extract_text_from_pdf(file_path="example.pdf", start_page=2))
