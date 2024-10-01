import chromadb
from langsmith import Client
from langsmith.evaluation import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_fireworks import FireworksEmbeddings, Fireworks
import os
from dotenv import load_dotenv
from langsmith.evaluation import LangChainStringEvaluator

# Cargar variables de entorno
load_dotenv()

# Configuración de la API key de LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Evaluator_v2"
if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = "fw_3ZJRGszUTX6LNDL7z6rEUUWB"

# Cliente LangSmith
client = Client()

# Cargar el modelo de Fireworks (Mistral)
model_wrapper = Fireworks(
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    temperature=0.6,
    max_tokens=400,
)

# Configura tu cliente de ChromaDB
CHROMA_PATH = "data/Test100"  # Actualiza esta ruta según tu entorno
DB_NAME = "test100"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=DB_NAME)

# Cargar el modelo para embeddings de Fireworks
embedding_model = FireworksEmbeddings(
    model="nomic-ai/nomic-embed-text-v1.5"
)

def retrieve_docs(question):
    """
    Recupera documentos relevantes de la base de datos ChromaDB usando el embedding de Fireworks.
    """
    # Genera el embedding de la consulta
    query_embedding = embedding_model.embed_documents([question])[0]

    # Realiza la consulta en la colección de ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=8
    )

    if not results['documents']:
        print("Unable to find matching results.")
        return ""

    # Extrae y formatea los textos recuperados
    context_texts = [item for sublist in results['documents'] for item in sublist]
    context_text = "\n\n---\n\n".join(context_texts)
    return context_text

class RagBot:
    def __init__(self, model, retriever):
        self._model = model
        self._retriever = retriever

    def get_answer(self, question: str):
        docs = self._retriever(question)
        prompt = f"""Basándote en la siguiente información, responde la pregunta de manera concisa, sé asertivo y amable.

Información:
{docs}

Pregunta: {question}

Respuesta:"""

        # Generar la respuesta usando el modelo
        result = self._model.generate([prompt])

        # Acceder al texto generado
        generated_text = result.generations[0][0].text

        return generated_text, docs  # Devuelve la respuesta y los documentos

rag_bot = RagBot(model_wrapper, retrieve_docs)

def predict_rag_answer_with_context(example):
    question = example["input_question"]
    response, contexts = rag_bot.get_answer(question)
    # Retornamos la predicción y el contexto utilizado
    return {"prediction": response, "context": contexts}

# Definir los criterios de evaluación
criteria = {
    "completeness": "La respuesta aborda todos los aspectos de la pregunta.",
    "relevance": "La respuesta es relevante a la pregunta y utiliza información del contexto proporcionado.",
    "correctness": "La respuesta es correcta y está libre de errores factuales.",
}

# Función prepare_data para mapear los campos correctamente
def prepare_data(run, example):
    return {
        "prediction": run.outputs['prediction'],
        "input": example.inputs['input_question'],
        "reference": example.outputs.get('output_answer', ''),
        "context": run.outputs.get('context', '')
    }

# Crear el evaluador utilizando LangChainStringEvaluator con 'context_qa'
context_qa_evaluator = LangChainStringEvaluator(
    evaluator="context_qa",
    config={
        "llm": model_wrapper
    },
    prepare_data=prepare_data
)

# Nombre del dataset existente en LangSmith
dataset_name = "v100-Eval dataset para RAG"

# Ejecutar la evaluación
experiment_results = evaluate(
    predict_rag_answer_with_context,
    data=dataset_name,
    evaluators=[context_qa_evaluator],
    experiment_prefix="rag-context-qa-v2",
    metadata={"version": "llama-v3p1-70b-instruct"}
)
