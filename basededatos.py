from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_fireworks import FireworksEmbeddings
import chromadb
import os
import shutil
from dotenv import load_dotenv
import json
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2  # PyPDF2 para la extracción de texto de PDFs

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
# Obtiene la api key de hugging face desde .env
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")

if "FIREWORKS_API_KEY" not in os.environ:
    os.environ["FIREWORKS_API_KEY"] = "fw_3ZJRGszUTX6LNDL7z6rEUUWB"
    
CHROMA_PATH = "data/Test100"  # Define la ruta de almacenamiento
DATA_PATH = "data"
DB_NAME = "test100"  # Nombre de la base de datos

# Tamaño del lote para el procesamiento
BATCH_SIZE = 200  # Ajusta este tamaño según el límite de tu modelo

# Inicializa el modelo de embeddings
embedding_model = FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")

def main():
    generate_data_store()

# Genera DB
def generate_data_store():
    documents = load_documents()  # Carga los documentos a utilizar
    chunks = split_text(documents)  # Divide los documentos en chunks
    save_to_chroma(chunks)  # Guarda los chunks en la DB

# Extrae el texto respetando la estructura de input/output
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

def load_json_files(directory, glob_pattern):
    json_files = glob.glob(os.path.join(directory, glob_pattern))
    documents = []
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Extrae el texto respetando la estructura de input/output
                text = extract_text_from_json(data)
                documents.append(Document(page_content=text, metadata={"source": file_path}))
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {file_path}")
    return documents

def extract_text_from_pdf(pdf_path):
    """
    Usa PyPDF2 para extraer el texto de un PDF.
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()  # Extrae el texto de cada página
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        text = ""
    return text

def load_documents():
    pdf_documents = []
    
    for pdf_file in glob.glob(os.path.join(DATA_PATH, "*.pdf")):
        text = extract_text_from_pdf(pdf_file)  # Extrae texto usando PyPDF2
        pdf_documents.append(Document(page_content=text, metadata={"source": pdf_file}))
    
    json_documents = load_json_files(DATA_PATH, "*.json")

    # Combina los documentos PDF y JSON
    documents = pdf_documents + json_documents
    print(f"Loaded {len(pdf_documents)} PDF documents and {len(json_documents)} JSON documents.")
    return documents

def split_text(documents: list[Document]):
    # Configura el RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: list[Document]):
    # Limpia la base de datos si llegase a existir algo
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Inicializa el cliente de Chroma con persistencia desde el principio
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Divide los chunks en lotes más pequeños
    def batch(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    # Procesa en lotes
    texts = [chunk.page_content for chunk in chunks]
    embeddings_list = []
    ids = [f"id_{i}" for i in range(len(texts))]

    for text_batch in batch(texts, BATCH_SIZE):
        try:
            # Usa el método adecuado para obtener los embeddings
            batch_embeddings = embedding_model.embed_documents(text_batch)
            embeddings_list.extend(batch_embeddings)
        except Exception as e:
            print(f"Error while embedding documents: {e}")

    # Crea una colección
    collection = chroma_client.create_collection(name=DB_NAME)

    # Inserta el documento en la colección
    collection.upsert(
        documents=texts,
        ids=ids,
        embeddings=embeddings_list
    )

    # Si se guardó correctamente lo imprime por pantalla
    print(f"Saved {len(chunks)} chunks to the collection.")

if __name__ == "__main__":
    main()
