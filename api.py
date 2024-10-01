from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from generarTexto import RagBot, model_wrapper, retrieve_docs

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],  # Cambia esto para restringir los orígenes permitidos
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)

rag_bot = RagBot(model_wrapper, retrieve_docs)

class Question(BaseModel):
    question: str

@app.post("/get_answer")
async def get_answer(question: Question):
    if not question.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    try:
        response, context = rag_bot.get_answer(question.question)
        return {"answer": response, "context": context}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la pregunta: {e}")