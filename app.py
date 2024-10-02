import streamlit as st
import requests

# Configura la URL de tu API de FastAPI
API_URL = "https://test-chatbot-z3yi.onrender.com"

# Definir estilos en Streamlit
st.set_page_config(page_title="UnoAfp GPT", page_icon="🤖", layout="centered")

# Estilos personalizados usando markdown y CSS
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #4a4a4a;
        text-align: center;
        margin-bottom: 20px;
    }
    .question-box {
        border: 2px solid #4a4a4a;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 20px;
    }
    .answer-box {
        background-color: #e0e0e0;
        color: black;
        border-left: 6px solid #888888;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .context-box {
        background-color: #e0e0e0;
        color: black;
        border-left: 6px solid #555555;
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        font-size: 14px;
        color: #7f8c8d;
    }
    </style>
""", unsafe_allow_html=True)

# Título estilizado
st.markdown('<div class="main-title">UnoAfp GPT 🤖</div>', unsafe_allow_html=True)

st.write("**Ingresa una pregunta y el modelo proporcionará una respuesta basada en el contexto recuperado.**")

# Caja de texto con estilos
question = st.text_area("Pregunta:", height=100, help="Escribe tu pregunta aquí", key="question")

# Botón para obtener respuesta
if st.button("Obtener Respuesta"):
    if question.strip():
        with st.spinner('Generando respuesta...'):
            try:
                # Hacer una solicitud POST al endpoint correcto de la API
                response = requests.post(f"{API_URL}/get_answer", json={"question": question})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Mostrar la respuesta con estilo
                    st.markdown(f'<div class="answer-box"><strong>Respuesta:</strong><br>{data.get("answer", "No se obtuvo respuesta.")}</div>', unsafe_allow_html=True)
                    
                    # Mostrar el contexto con estilo
                    st.markdown(f'<div class="context-box"><strong>Contexto Utilizado:</strong><br>{data.get("context", "No se obtuvo contexto.")}</div>', unsafe_allow_html=True)
                else:
                    error_detail = response.json().get('detail', 'Error desconocido.')
                    st.error(f"Error {response.status_code}: {error_detail}")
            
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API. Asegúrate de que el backend está corriendo.")
            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
    else:
        st.warning("Por favor, ingresa una pregunta válida.")

# Pie de página
st.markdown('<div class="footer">Desarrollado por Platwave | 2024</div>', unsafe_allow_html=True)
