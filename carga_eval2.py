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
dataset_name = "v100-Eval dataset para RAG"
dataset = client.create_dataset(
    dataset_name,
    description="Dataset para evaluación de modelo RAG con Llama"
)

# Crea ejemplos dentro del dataset
client.create_examples(
    inputs=[
    {"input_question": "Estoy planeando mi jubilación, ¿cómo puede el enfoque de bajo costo de uno afp ayudarme a obtener una mejor pensión en 2023?"},
    {"input_question": "Me preocupa el impacto ambiental de mis inversiones, ¿qué está haciendo uno afp en 2023 para invertir de manera sostenible?"},
    {"input_question": "Si la economía global enfrenta una recesión, ¿cómo protege uno afp mis ahorros para la jubilación en 2023?"},
    {"input_question": "Quiero entender mejor mis opciones de inversión, ¿qué recursos educativos ofrece uno afp para clientes como yo en 2023?"},
    {"input_question": "¿Cómo reflejan los valores de integridad y adaptabilidad de uno afp la forma en que manejan las tendencias emergentes en el sector financiero en 2023?"}
],
    outputs=[
    {"output_answer": "El enfoque de bajo costo de uno afp significa que cobramos comisiones más bajas, lo que permite que una mayor parte de tus aportes se invierta en tu fondo de pensiones. A lo largo del tiempo, esto puede incrementar el rendimiento de tus ahorros y resultará en una mejor pensión al momento de tu jubilación."},
    {"output_answer": "En uno afp, estamos comprometidos con la inversión sostenible. En 2023, incorporamos criterios ambientales, sociales y de gobernanza (ESG) en nuestras decisiones de inversión para apoyar proyectos y empresas que tienen un impacto positivo en el medio ambiente."},
    {"output_answer": "Uno afp implementa estrategias de diversificación y gestión de riesgos para minimizar el impacto de las fluctuaciones del mercado en tus ahorros. Nuestro equipo de expertos monitorea constantemente el mercado para ajustar las inversiones y proteger tus fondos durante períodos de inestabilidad económica."},
    {"output_answer": "En 2023, uno afp ofrece una variedad de recursos educativos, incluyendo seminarios web, talleres y herramientas en línea, para ayudarte a comprender tus opciones de inversión y tomar decisiones informadas sobre tus ahorros para la jubilación."},
    {"output_answer": "Nuestros valores de integridad y adaptabilidad nos permiten responder eficazmente a las tendencias emergentes en el sector financiero. En 2023, estamos adoptando nuevas tecnologías y prácticas innovadoras para mejorar nuestros servicios y garantizar que tus inversiones estén alineadas con las mejores oportunidades del mercado."}
    ],
    dataset_id=dataset.id,
)

