import os
import aiofiles
import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pdf_to_db import PDFVecDataBase
from dotenv import load_dotenv
from reranker import Rerank
from llm_model import LLMModel

# Загружаем переменные из .env файла
load_dotenv()

# Получаем переменные
EMBEDDINGS_MODEL = os.getenv('EMBEDDINGS_MODEL')
RERANK_MODEL = os.getenv('RERANK_MODEL')
LLM_MODEL = os.getenv('LLM_MODEL')
PATH_DB = os.getenv('PATH_DB')
UPLOAD_DIR = os.getenv('UPLOAD_DIR')

pdf_db = PDFVecDataBase(embeddings_model=EMBEDDINGS_MODEL,
                        path_db=PATH_DB)

reranker = Rerank(RERANK_MODEL, top_n=5)

llm_model = LLMModel(LLM_MODEL)
llm_model.load_model()

splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500,
                                          separators=["\n\n", "\n", ",", " ", ""])

upload_dir = Path(UPLOAD_DIR)
upload_dir.mkdir(exist_ok=True)

app = FastAPI()


@dataclass
class UserRequest:
    collection_name: str
    question: str


@dataclass
class UserResponse:
    answer: str


@app.post("/add_pdf_to_db")
async def add_pdf_to_db(collection_name: str = Form(..., description="Название коллекции"),
                        start_page: int = Form(1, description="Страница, с которой начать обработку (по умолчанию 1)"),
                        overwrite: bool = Form(False,
                                               description="Перезаписать коллекцию если существует (по умолчанию False)"),
                        file: UploadFile = File(..., description="PDF файл для загрузки")) -> dict[str, str]:
    if not file:
        raise HTTPException(status_code=400, detail="Файл не передан")

    filename = file.filename

    # Проверка расширения файла (опционально)
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail=f"Файл {filename} не имеет расширения .pdf")

    file_path = upload_dir / filename

    # Асинхронное сохранение файла
    try:
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await file.read()
            await buffer.write(content)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Файл {filename} не загружен: {e}")

    try:
        pdf_db.add_pdf_to_db(file_path=str(file_path), collection_name=collection_name, text_splitter=splitter,
                             start_page=start_page, overwrite=overwrite)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка на этапе извлечения текста из PDF файла и добавления в "
                                                    f"векторную базу данных: {e}")

    if overwrite:
        return {
            "message": f"Коллекция '{collection_name}' перезаписана. Файл {filename} загружен в коллекцию",
            "collection_name": collection_name,
            "filename": filename,
            "status": "success",
            "action": "overwritten"
        }
    return {
        "message": f"Файл {filename} загружен в коллекцию '{collection_name}'",
        "collection_name": collection_name,
        "filename": filename,
        "status": "success",
        "action": "added"
    }


@app.get("/get_existing_collections")
async def get_existing_collections() -> dict[str, list[str]]:
    existing_collections = pdf_db.list_collection()

    return {
        "existing_collections": existing_collections,
    }


@app.post("/delete_collection")
async def delete_collection(collection_name: str = Form(..., description="Название коллекции")) -> dict[str, str]:
    try:
        pdf_db.delete_collection(collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Ошибка удаления коллекции '{collection_name}' из векторной базы данных: {e}")

    return {
        "message": f"Коллекция '{collection_name}' удалена из векторной базы данных",
        "status": "success",
        "action": "delete"
    }


@app.post("/question")
async def answers_questions(data: UserRequest) -> UserResponse:
    try:
        collection_name = data.collection_name
        question = data.question

        if not collection_name:
            raise HTTPException(status_code=422, detail="Отсутствует название коллекции")
        if not question:
            raise HTTPException(status_code=422, detail="Отсутствует вопрос")

        existing_collections = pdf_db.list_collection()
        if collection_name not in existing_collections:
            raise HTTPException(status_code=400, detail=f"Отсутствует коллекция с названием '{collection_name}'")

        collection_documents = pdf_db.load_collection(collection_name=collection_name).invoke(question)

        second_docs = reranker.compress_documents(query=question, documents=collection_documents)

        separator = "\n===========\n"
        context = separator.join(doc.page_content for doc in second_docs)

        answer = llm_model.answer_question(query=question, context=context)

        return UserResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка выполнения запроса: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.10.169", port=8080)
