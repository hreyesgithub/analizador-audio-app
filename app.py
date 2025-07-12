import streamlit as st
import whisper
import cohere
import os # Importamos el módulo 'os' para manejar archivos

# --- CONFIGURACIÓN ---
# Usa los "secrets" de Streamlit para la clave de API
try:
    COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
    co = cohere.Client(COHERE_API_KEY)
except Exception as e:
    st.error(f"Error al configurar la API Key de Cohere. Asegúrate de haberla añadido en los 'Secrets' de tu app en Streamlit Cloud. Error: {e}")
    st.stop()

# --- TÍTULO Y DESCRIPCIÓN ---
st.set_page_config(page_title="Asistente de Análisis de Audio", layout="wide")
st.title("🚀 Asistente de Análisis de Audio")
st.markdown("Sube un archivo de audio (MP3, WAV, M4A) y obtén una transcripción, resumen y análisis detallado.")

# --- FUNCIONES DE ANÁLISIS ---
@st.cache_data
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    try:
        result = model.transcribe(file_path, fp16=False) # fp16=False puede mejorar la compatibilidad
        return result
    except Exception as e:
        st.error(f"Error durante la transcripción con Whisper: {e}")
        return None

@st.cache_data
def get_cohere_analysis(_client, prompt, text_to_analyze):
    try:
        response = _client.chat(model="command-r", message=f"{prompt}\n\n---\n{text_to_analyze}\n---", temperature=0.2)
        return response.text
    except Exception as e:
        return f"Error en la llamada a la API de Cohere: {e}"

# --- INTERFAZ DE USUARIO ---
uploaded_file = st.file_uploader("Elige un archivo de audio", type=['mp3', 'wav', 'm4a', 'm4b', 'mp4', 'mpeg'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # --- CAMBIO CLAVE: Guardar el archivo temporalmente ---
    # Creamos una ruta temporal para el archivo
    temp_file_path = os.path.join(".", uploaded_file.name)
    
    # Leemos los bytes del archivo subido y los escribimos en el disco del servidor
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # ---------------------------------------------------------

    with st.spinner('Transcripción en progreso... Esto puede tardar varios minutos.'):
        # Pasamos la RUTA COMPLETA del archivo guardado a Whisper
        transcription_result = transcribe_audio(temp_file_path)

    # --- CAMBIO CLAVE: Limpiar el archivo temporal ---
    os.remove(temp_file_path)
    # ----------------------------------------------------

    if transcription_result:
        full_text = transcription_result['text']
        st.success("Transcripción completada.")

        # Realizar análisis con Cohere
        st.header("📊 Análisis de la Reunión")
        
        with st.spinner("Generando resumen y lista de tareas..."):
            st.subheader("✍️ Resumen y Tareas")
            prompt_summary_tasks = "Genera un resumen conciso de la siguiente transcripción, seguido de una lista de tareas (To-Do list) con viñetas y responsables si se mencionan."
            summary_tasks = get_cohere_analysis(co, prompt_summary_tasks, full_text)
            st.markdown(summary_tasks)
        
        with st.spinner("Analizando sentimiento y entidades..."):
            st.subheader("😊 Sentimiento y Entidades Clave")
            prompt_sentiment_entities = "Analiza el sentimiento general (Positivo, Negativo o Neutral) y extrae las personas y empresas clave mencionadas en la transcripción."
            sentiment_entities = get_cohere_analysis(co, prompt_sentiment_entities, full_text)
            st.info(sentiment_entities)

        with st.expander("📜 Ver Transcripción Completa"):
            st.text_area("Transcripción", full_text, height=300)