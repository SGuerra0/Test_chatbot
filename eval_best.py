import chromadb
from langsmith import Client
from langsmith.evaluation import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import PromptTemplate
from langchain_fireworks import FireworksEmbeddings, Fireworks
import os
from dotenv import load_dotenv

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
    temperature=0.3,
    max_tokens=400,
    
)

# Configura tu cliente de ChromaDB
CHROMA_PATH = "data/Test100"  # Actualiza esta ruta
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
        prompt = f"""Basándote en la siguiente información, responde la pregunta de manera concisa, se asertivo y amable

        Información:
        {docs}

        Pregunta: {question}

        Respuesta:"""

        # Convertir el prompt a cadena
        result = self._model.generate([prompt])

        # Acceder al texto generado desde el LLMResult
        generated_text = result.generations[0][0].text

        return generated_text, docs  # Devuelve la respuesta y los documentos

rag_bot = RagBot(model_wrapper, retrieve_docs)

def predict_rag_answer_with_context(example):
    """
    Predice la respuesta basada en la pregunta de ejemplo y los contextos recuperados.
    """
    question = example["input_question"]
    response, contexts = rag_bot.get_answer(question)
    return {"answer": response, "contexts": contexts}

def docs_relevance_evaluator(run, example):
    """
    Evalúa la relevancia de los documentos recuperados para la pregunta dada.
    """
    input_question = example["input_question"] if "input_question" in example else ""
    
    # Extraer contextos de los resultados
    contexts = run.outputs["contexts"] if hasattr(run, 'outputs') and "contexts" in run.outputs else ""

    prompt = f"""Evalúa la relevancia de los siguientes documentos recuperados para la pregunta dada.
    Proporciona una puntuación del 0 al 10, donde 0 es completamente irrelevante y 10 es altamente relevante.
    Pregunta: {input_question}

    Documentos recuperados:
    {contexts}

    Evaluación:"""

    # Convertir el prompt a cadena
    result = model_wrapper.generate([prompt])

    # Acceder al texto generado desde el LLMResult
    generated_text = result.generations[0][0].text
    print("Resultado de la generación:", generated_text)

    # Extraer la puntuación del resultado
    try:
        score_text = generated_text.split("Puntuación:")[-1].strip().split()[0]
        score = float(score_text.replace(',', '.'))  # Reemplaza coma con punto si es necesario
        normalized_score = score / 10.0  # Normalizar a rango 0-1
    except Exception as e:
        print(f"Error al extraer la puntuación: {e}")
        normalized_score = 0.0  # Valor por defecto si no se puede extraer la puntuación

    return {"key": "document_relevance", "score": normalized_score, "comment": generated_text}

# Nombre del dataset existente en LangSmith
dataset_name = "v100-Eval dataset para RAG"

# Ejecutar la evaluación
experiment_results = evaluate(
    predict_rag_answer_with_context,
    data=dataset_name,  # Reemplaza esto con tu conjunto de datos real
    evaluators=[docs_relevance_evaluator],
    experiment_prefix="rag-doc-relevance-v2",
    metadata={"version": "llama-v3p1-70b-instruct"}
)
