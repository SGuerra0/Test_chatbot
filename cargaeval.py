from langsmith import Client

import os
# Configuración de la API key de LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Evaluator_v2"

# Cliente LangSmith
client = Client()
# Definimos el dataset a crear
dataset_name = "v2-Eval dataset para RAG"
dataset = client.create_dataset(
    dataset_name,
    description="Dataset para evaluación de modelo RAG con Llama"
)

# Crea ejemplos dentro del dataset
client.create_examples(
    inputs=[
        {"input_question": "Quienes son los integrantes del directorio de uno afp en 2023?"},
        {"input_question": "Quienes son los directores suplentes de uno afp en 2023?"},
        {"input_question": "Cuál es el propósito de uno afp en 2023?"},
        {"input_question": "Cuál es la misión de uno afp en 2023?"},
        {"input_question": "Cuáles son los valores de uno afp en 2023?"},
        {"input_question": "Cuál es la política de inversión y financiamiento de uno afp en 2023?"},
        {"input_question": "¿Qué es la ley de protección de empleos?"}
    ],
    outputs=[
        {"output_answer": "Los integrantes del directorio en 2023 son: Mario Ignacio Alvarez Avendaño, Hugo Felipe Ovando Zalazar, Felipe Eduardo Aldunate Anfossi, Pablo Andres Arze Romani y Claudia Verdugo Celedon"},
        {"output_answer": "Los directores suplentes son Monserrat Nova Radic y Santiago Truffa Sotomayor"},
        {"output_answer": "El propósito de uno afp es: Nos mueve mejorar la vida de todos hoy y mañana. Por eso te cobramos lo menos posible para que tú ganes más"},
        {"output_answer": "La misión de uno afp es: Nos esforzamos en tener el menor costo para que nuestros afiliados ganen más, beneficiando así a los trabajadores y sus familias que con esfuerzo ahorran todos los meses para su jubilación."},
        {"output_answer": "Los valores de uno afp son: La integridad, excelencia, adaptabilidad e innovación"},
        {"output_answer": "La política de inversión y financiamiento es: La principal inversión de la Administradora se relaciona al encaje, establecido por ley en el D.L. 3500, que corresponde al 1% de los recursos administrados en cada fondo de pensiones, el cual se invierte en cuotas de los mismos fondos, estando por tanto sujeto a la misma regulación y compartiendo los resultados de éstos."},
        {"output_answer": "¿Qué es la Ley de Protección al Empleo?Es un beneficio estatal que ayudará a proteger los empleos frente a esta crisis sanitaria que estamos enfrentando. Se aplica bajo 3 causales y permite la activación del seguro de cesantía entre otras cosas."}
    ],
    dataset_id=dataset.id,
)

