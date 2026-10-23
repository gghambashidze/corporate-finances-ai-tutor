import os
import json
import uuid
import tempfile
import streamlit as st
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from gtts import gTTS

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CONFIG & PERSISTENCE ---
st.set_page_config(page_title="Corporate Finances", layout="wide")

SESSIONS_FILE = "chat_sessions.json"
PROFILE_FILE = "user_profile.json"
HOMEWORK_DIR = "./saved_homework"

if not os.path.exists(HOMEWORK_DIR):
    os.makedirs(HOMEWORK_DIR)

def load_json(filepath, default):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

def render_financial_chart(chart_type, labels, values, title="Corporate Finance Analysis"):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('#444444')

    if chart_type == "bar":
        bars = ax.bar(labels, values, color='#4F8BF9')
        ax.bar_label(bars, fmt='%.1f', padding=3, color='white')
    elif chart_type == "pie":
        ax.pie(values, labels=labels, autopct='%1.1f%%', textprops={'color':"w"})
    elif chart_type == "line":
        ax.plot(labels, values, marker='o', color='#4F8BF9', linewidth=2)
        ax.grid(True, linestyle='--', alpha=0.3)

    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)

# --- SESSION & PROFILE INITIALIZATION ---
raw_sessions = load_json(SESSIONS_FILE, {"default": {"name": "Default Thread", "messages": []}})
valid_sessions = {}
for sid, data in raw_sessions.items():
    if isinstance(data, dict):
        data.setdefault("name", f"Thread {sid}")
        data.setdefault("messages", [])
        valid_sessions[sid] = data
    elif isinstance(data, list):
        valid_sessions[sid] = {"name": f"Thread {sid}", "messages": data}

if not valid_sessions:
    valid_sessions = {"default": {"name": "Default Thread", "messages": []}}

if "sessions" not in st.session_state:
    st.session_state.sessions = valid_sessions

if "active_session_id" not in st.session_state or st.session_state.active_session_id not in st.session_state.sessions:
    st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]

if "profile" not in st.session_state:
    st.session_state.profile = load_json(PROFILE_FILE, {"learnt_material": []})

def generate_voice_response(text):
    try:
        clean_text = text.replace("#", "").replace("*", "").replace("`", "").strip()
        if not clean_text:
            return None
        if len(clean_text) > 500:
            clean_text = clean_text[:500] + "..."
        tts = gTTS(text=clean_text, lang="en")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_audio.name)
        return temp_audio.name
    except Exception:
        return None

# --- RAG & LLM INITIALIZATION ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = Ollama(model="llama3.2:1b")

TEACH_PROMPT = ChatPromptTemplate.from_template("""
You are an expert, direct, and concise Corporate Finance AI Tutor.

Context from Course Material:
{context}

Learnt Material History:
{learnt_material}

Student Input:
{user_input}

Instructions:
1. Provide a direct, concise answer (under 150 words).
2. If grading math or student work, verify step-by-step arithmetic independently.
3. Clearly state whether the concept or calculation is Correct or Incorrect.
4. Keep explanations brief and avoid unnecessary preamble.
""")

# --- SIDEBAR 1: THREAD MANAGER ---
st.sidebar.title("Navigation & Threads")
if st.sidebar.button("New Chat Thread", use_container_width=True):
    new_id = str(uuid.uuid4())[:8]
    st.session_state.sessions[new_id] = {"name": f"Thread {new_id}", "messages": []}
    st.session_state.active_session_id = new_id
    save_json(SESSIONS_FILE, st.session_state.sessions)
    st.rerun()

session_options = {sid: data.get("name", f"Thread {sid}") for sid, data in st.session_state.sessions.items()}
active_idx = list(session_options.keys()).index(st.session_state.active_session_id)

selected_session = st.sidebar.selectbox(
    "Active Thread",
    options=list(session_options.keys()),
    format_func=lambda x: session_options[x],
    index=active_idx
)
st.session_state.active_session_id = selected_session

with st.sidebar.expander("Manage Selected Thread"):
    current_name = st.session_state.sessions[selected_session].get("name", f"Thread {selected_session}")
    new_name = st.text_input("Thread Name", value=current_name)
    
    col_rename, col_delete = st.columns(2)
    with col_rename:
        if st.button("Rename"):
            if new_name.strip():
                st.session_state.sessions[selected_session]["name"] = new_name.strip()
                save_json(SESSIONS_FILE, st.session_state.sessions)
                st.toast("Thread renamed successfully")
                st.rerun()

    with col_delete:
        if st.button("Delete Thread"):
            if len(st.session_state.sessions) > 1:
                del st.session_state.sessions[selected_session]
                st.session_state.active_session_id = list(st.session_state.sessions.keys())[0]
                save_json(SESSIONS_FILE, st.session_state.sessions)
                st.toast("Thread deleted")
                st.rerun()
            else:
                st.warning("Cannot delete the last remaining thread.")

# --- SIDEBAR 2: LEARNT MATERIAL ---
st.sidebar.markdown("---")
st.sidebar.subheader("Learnt Material Explorer")
if st.session_state.profile.get("learnt_material"):
    for concept in st.session_state.profile["learnt_material"]:
        st.sidebar.markdown(f"- {concept}")
else:
    st.sidebar.info("No concepts logged yet.")

# --- SIDEBAR 3: HOMEWORK CAMERA & FILE STORAGE ---
st.sidebar.markdown("---")
st.sidebar.subheader("Homework Camera Grader")

cam_tab, upload_tab = st.sidebar.tabs(["Webcam", "Upload Image"])

captured_img = None
with cam_tab:
    cam_file = st.camera_input("Take photo of homework")
    if cam_file: 
        captured_img = Image.open(cam_file)

with upload_tab:
    up_file = st.file_uploader("Upload assignment file", type=["png", "jpg", "jpeg"])
    if up_file: 
        captured_img = Image.open(up_file)

if captured_img is not None:
    st.sidebar.image(captured_img, caption="Scanned Homework", use_container_width=True)
    
    col_check, col_save = st.sidebar.columns(2)
    
    with col_save:
        if st.button("Save File"):
            filename = f"homework_{uuid.uuid4().hex[:6]}.png"
            filepath = os.path.join(HOMEWORK_DIR, filename)
            captured_img.save(filepath)
            st.toast(f"Saved homework as {filename}")

    with col_check:
        if st.button("Check Image"):
            with st.spinner("Extracting handwritten/printed text locally..."):
                try:
                    ocr_text = pytesseract.image_to_string(captured_img)
                except Exception:
                    ocr_text = ""
                    st.sidebar.error("OCR Engine error. Please verify tesseract-ocr is installed.")
            
            if not ocr_text.strip():
                st.sidebar.error("Could not extract text clearly. Ensure good lighting and focus.")
            else:
                active_id = st.session_state.active_session_id
                user_msg = f"[HOMEWORK SUBMISSION]:\n{ocr_text}"
                
                st.session_state.sessions[active_id]["messages"].append({"role": "user", "content": user_msg})
                
                docs = retriever.invoke(user_msg)
                context = "\n\n".join(doc.page_content for doc in docs)
                tutor_chain = TEACH_PROMPT | llm | StrOutputParser()
                
                with st.spinner("Evaluating image content..."):
                    response = tutor_chain.invoke({
                        "learnt_material": ", ".join(st.session_state.profile.get("learnt_material", [])) or "None",
                        "context": context,
                        "user_input": user_msg
                    })
                
                st.session_state.sessions[active_id]["messages"].append({"role": "assistant", "content": response})
                save_json(SESSIONS_FILE, st.session_state.sessions)
                st.rerun()

with st.sidebar.expander("Saved Homework Files"):
    saved_files = os.listdir(HOMEWORK_DIR) if os.path.exists(HOMEWORK_DIR) else []
    if saved_files:
        for fname in saved_files:
            fpath = os.path.join(HOMEWORK_DIR, fname)
            
            with st.expander(fname):
                if os.path.exists(fpath):
                    saved_img = Image.open(fpath)
                    st.image(saved_img, caption=fname, use_container_width=True)
                    if st.button("Delete File", key=f"del_{fname}"):
                        os.remove(fpath)
                        st.toast(f"Deleted {fname}")
                        st.rerun()
    else:
        st.write("No saved homework files.")

# --- MAIN CHAT INTERFACE ---
st.title("Corporate Finances AI Tutor")

active_id = st.session_state.active_session_id
current_messages = st.session_state.sessions[active_id]["messages"]

for idx, msg in enumerate(current_messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if st.button("Read Out Loud", key=f"audio_{idx}"):
                audio_file = generate_voice_response(msg["content"])
                if audio_file:
                    st.audio(audio_file, format="audio/mp3", autoplay=True)

user_input = st.chat_input("Ask a question, submit homework, or request a chart...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.sessions[active_id]["messages"].append({"role": "user", "content": user_input})
    
    docs = retriever.invoke(user_input)
    context = "\n\n".join(doc.page_content for doc in docs)
    tutor_chain = TEACH_PROMPT | llm | StrOutputParser()
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = tutor_chain.invoke({
                "learnt_material": ", ".join(st.session_state.profile.get("learnt_material", [])) or "None",
                "context": context,
                "user_input": user_input
            })
            st.markdown(response)
            
            if any(k in user_input.lower() for k in ["chart", "graph", "plot"]):
                st.subheader("Dynamic Financial Visualization")
                render_financial_chart(
                    chart_type="pie", 
                    labels=["Equity (60%)", "Debt (40%)"], 
                    values=[60, 40], 
                    title="Capital Structure Breakdown"
                )

    st.session_state.sessions[active_id]["messages"].append({"role": "assistant", "content": response})
    save_json(SESSIONS_FILE, st.session_state.sessions)
