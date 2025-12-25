import uvicorn
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException

app = FastAPI()

@dataclass
class UserRequest:
    question: str

@dataclass
class UserResponse:
    answer: str

@app.post("/question")
async def answers_questions(data: UserRequest) -> UserResponse:

    try:
        question = data.question

        if not question:
            raise HTTPException(status_code=422, detail="Отсутствует вопрос")
        else:
            return UserResponse(answer="Good job!")

    except Exception as e:
      raise HTTPException(status_code=400, detail=f"Ошибка выполнения запроса: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="192.168.10.148", port=8080)