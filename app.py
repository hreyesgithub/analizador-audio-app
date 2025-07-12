import streamlit as st
import whisper
import cohere
import os
from pydub import AudioSegment

# --- CONFIGURACIÓN ---
# Se usan los "secrets" de Streamlit para mayor seguridad
COHERE_API_KEY = st.secrets["wlYgaiR7hrfpX5tit2gRhTsbwHaynpZ8KBu3S68H"]
co = cohere.Client(COHERE_API_KEY)

# --- TÍTULO Y DESCRIPCIÓN ---
st.set_page_config(page_title="Asistente de Análisis de Audio", layout="wide")
st.title("🚀 Asistente de Análisis de Audio")
st.markdown("Sube un archivo de audio (MP3, WAV, etc.) y obtén una transcripción, resumen y análisis detallado.")

# --- FUNCIONES DE ANÁLISIS ---
@st.cache_data
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result

@st.cache_data
def get_cohere_analysis(_client, prompt, text_to_analyze):
    response = _client.chat(model="command-r", message=f"{prompt}\n\n---\n{text_to_analyze}\n---", temperature=0.2)
    return response.text

# --- INTERFAZ DE USUARIO ---
uploaded_file = st.file_uploader("Elige un archivo de audio", type=['mp3', 'wav', 'm4a', 'ogg'])

if uploaded_file is not None:
    # Guardar archivo temporalmente
    with st.spinner('Procesando audio... por favor, espera.'):
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Archivo '{file_path}' subido con éxito.")

        # Transcripción
        transcription_result = transcribe_audio(file_path)
        full_text = transcription_result['text']

        # Análisis con Cohere
        st.header("📊 Análisis de la Reunión")
        
        # Usar columnas para un diseño más limpio
        col1, col2 = st.columns(2)

        with col1:
            with st.spinner("Generando resumen y lista de tareas..."):
                st.subheader("✍️ Resumen y Tareas")
                prompt_summary_tasks = "Genera un resumen conciso de la siguiente transcripción, seguido de una lista de tareas (To-Do list) con viñetas y responsables si se mencionan."
                summary_tasks = get_cohere_analysis(co, prompt_summary_tasks, full_text)
                st.markdown(summary_tasks)
        
        with col2:
            with st.spinner("Analizando sentimiento y entidades..."):
                st.subheader("😊 Sentimiento General")
                prompt_sentimiento = "Analiza el sentimiento general de la transcripción. Responde solo con: Positivo, Negativo o Neutral y justifica brevemente."
                sentimiento = get_cohere_analysis(co, prompt_sentimiento, full_text)
                st.info(sentimiento)

                st.subheader("🏷️ Entidades Clave")
                prompt_entidades = "Extrae y lista las personas, fechas y empresas clave mencionadas."
                entidades = get_cohere_analysis(co, prompt_entidades, full_text)
                st.success(entidades)

        # Transcripción completa en un expander
        with st.expander("📜 Ver Transcripción Completa"):
            st.text_area("Transcripción", full_text, height=300)
        
        # Limpieza
        os.remove(file_path)