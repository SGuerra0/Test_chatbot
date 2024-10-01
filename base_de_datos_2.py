import os
from dotenv import load_dotenv
from langchain_fireworks import FireworksEmbeddings
from langchain.schema import Document
import chromadb
import json
import glob
import pypdfium2
import spacy
import uuid
from pydantic import BaseModel, Field

# Cargar el modelo de spaCy para español
nlp = spacy.load('es_core_news_sm')

# Descargar recursos de NLTK si aún no lo has hecho
import nltk
nltk.download('punkt')

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

# Configura tu cliente de ChromaDB
CHROMA_PATH = "data/Test100"
DB_NAME = "test100"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# Cargar el modelo para embeddings usando FireworksEmbeddings
embedding_model = FireworksEmbeddings(
    model="nomic-ai/nomic-embed-text-v1.5",
)

DATA_PATH = "data"

# Funciones para cargar y procesar documentos
def load_documents():
    # Cargar documentos PDF usando pypdfium2
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
    pdf_documents = []
    for pdf_file in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_file)
        if pdf_text.strip():
            pdf_documents.append(Document(page_content=pdf_text, metadata={"source": pdf_file}))
        else:
            print(f"No se extrajo texto del archivo PDF: {pdf_file}")

    # Cargar documentos JSON y extraer el contenido de input y output
    json_files = glob.glob(os.path.join(DATA_PATH, "*.json"))
    json_documents = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Extraer el texto de 'input' y 'output'
                text = extract_text_from_json(data)
                if text.strip():
                    json_documents.append(Document(page_content=text, metadata={"source": json_file}))
                else:
                    print(f"No se extrajo texto del archivo JSON: {json_file}")
            except json.JSONDecodeError:
                print(f"Error al decodificar JSON del archivo: {json_file}")

    documents = pdf_documents + json_documents
    print(f"Cargados {len(pdf_documents)} documentos PDF y {len(json_documents)} documentos JSON.")
    return documents

def extract_text_from_json(data):
    """
    Extrae los valores de 'input' como títulos y 'output' como contenido.
    """
    if isinstance(data, list):  # Asumimos que los registros están en una lista
        documents = []
        for item in data:
            if "input" in item and "output" in item:
                input_text = item.get("input", "")
                output_text = item.get("output", "")
                # Combina el input como título y el output como contenido
                full_text = f"Título: {input_text}\n\nContenido: {output_text}"
                documents.append(full_text)
        return "\n".join(documents)
    else:
        return ""

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_document = pypdfium2.PdfDocument(pdf_file)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            page_text = page.get_textpage().get_text_range()
            if page_text:
                text += page_text + "\n"
            else:
                print(f"No se extrajo texto de la página {page_num} del archivo PDF: {pdf_file}")
        pdf_document.close()
    except Exception as e:
        print(f"Error al abrir el archivo PDF: {pdf_file}. Error: {e}")
    return text

def split_text(documents, max_chunk_size=1000):
    """
    Divide los documentos en chunks semánticos utilizando spaCy.
    """
    chunks = []
    for doc in documents:
        text = doc.page_content
        spacy_doc = nlp(text)
        sentences = [sent.text.strip() for sent in spacy_doc.sents]

        current_chunk = ''
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length <= max_chunk_size:
                current_chunk += sentence + ' '
                current_length += sentence_length
            else:
                # Agregar el chunk actual a la lista de chunks
                chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))
                # Iniciar un nuevo chunk
                current_chunk = sentence + ' '
                current_length = sentence_length

        # Agregar el último chunk
        if current_chunk:
            chunks.append(Document(page_content=current_chunk.strip(), metadata=doc.metadata))

    print(f"Divididos {len(documents)} documentos en {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    # Preparar datos
    texts = [chunk.page_content for chunk in chunks if chunk.page_content.strip()]
    metadatas = [chunk.metadata for chunk in chunks if chunk.page_content.strip()]
    ids = [str(uuid.uuid4()) for _ in range(len(texts))]

    # Crear o obtener la colección
    if DB_NAME in [collection.name for collection in chroma_client.list_collections()]:
        collection = chroma_client.get_collection(name=DB_NAME)
    else:
        collection = chroma_client.create_collection(name=DB_NAME)

    # Procesar en lotes
    BATCH_SIZE = 200
    def batch(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    for texts_batch, metadatas_batch, ids_batch in zip(batch(texts, BATCH_SIZE), batch(metadatas, BATCH_SIZE), batch(ids, BATCH_SIZE)):
        embeddings_batch = embedding_model.embed_documents(texts_batch)
        collection.upsert(
            documents=texts_batch,
            metadatas=metadatas_batch,
            ids=ids_batch,
            embeddings=embeddings_batch
        )

    # Configura el cliente para que persista la base de datos en la ruta especificada
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    print(f"Guardados {len(texts)} chunks en la colección.")

def main():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()
