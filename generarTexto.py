# generarTexto.py

import os
from dotenv import load_dotenv
from langchain_fireworks import FireworksEmbeddings, Fireworks
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import chromadb
import nltk

# Descargar recursos de NLTK si aún no lo has hecho
nltk.download('punkt')

# Cargar variables de entorno
load_dotenv()

# Leer variables de entorno desde el archivo .env
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = FIREWORKS_API_KEY



# Cargar el modelo de Fireworks (Llama)
model_wrapper = Fireworks(
    model="accounts/fireworks/models/firefunction-v2",
    temperature=0.3,
    max_tokens=400
)

# Configura tu cliente de ChromaDB
CHROMA_PATH = "data/Test100"  # Actualiza esta ruta si es necesario
DB_NAME = "test100"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=DB_NAME)

# Cargar el modelo para embeddings usando FireworksEmbeddings
embedding_model = FireworksEmbeddings(
    model="nomic-ai/nomic-embed-text-v1.5",
)

def retrieve_docs(question):
    try:
        query_embedding = embedding_model.embed_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=8
        )
    except Exception as e:
        print(f"Error al recuperar documentos: {e}")
        return ""

    if not results['documents']:
        print("No se encontraron resultados coincidentes.")
        return ""

    # Filtra documentos por umbral de similitud
    threshold = 1.0  # Ajusta este valor según tus necesidades
    filtered_docs = []
    for doc_list, distance_list in zip(results['documents'], results['distances']):
        for doc, distance in zip(doc_list, distance_list):
            if distance >= threshold:
                filtered_docs.append(doc)

    # Si no hay documentos que cumplan el umbral, utiliza todos los documentos
    if not filtered_docs:
        filtered_docs = [item for sublist in results['documents'] for item in sublist]

    context_text = "\n\n---\n\n".join(filtered_docs)
    return context_text

class RagBot:
    def __init__(self, model, retriever):
        self._model = model
        self._retriever = retriever

    def get_answer(self, question: str):
        docs = self._retriever(question)
        if not docs.strip():
            return "No se pudo recuperar información relevante.", ""

        prompt = f"""Eres un experto en AFP Uno. Basándote en la siguiente información y en tus conocimientos, responde la pregunta de manera concisa, asertiva y amable. Utiliza la información proporcionada y realiza inferencias lógicas cuando sea necesario, no menciones si la información proporcionada no da detalles especificos.

        Información:    
        {docs}

        Pregunta: {question}

        Respuesta:"""


        try:
            result = self._model.generate([prompt])
            generated_text = result.generations[0][0].text.strip()
        except Exception as e:
            print(f"Error al generar la respuesta: {e}")
            generated_text = "Error al generar la respuesta."

        return generated_text, docs