"""
streamlit_app.py - Web UI for the Autonomous Llama Agent
Run: streamlit run streamlit_app.py
"""

import streamlit as st
from agent import build_agent, TOOLS, extract_output
from logger import SessionLogger
import config, time, uuid
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="Autonomous Llama Agent",
    page_icon="🦙",
    layout="wide",
)

st.markdown("""
<style>
.agent-action { background:#fff8ec; border-left:3px solid #EF9F27; padding:8px 12px; border-radius:0 6px 6px 0; font-size:13px; margin:4px 0; }
.agent-obs { background:#f0faf5; border-left:3px solid #1D9E75; padding:8px 12px; border-radius:0 6px 6px 0; font-size:13px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
   
    # Model Selection for Groq
    model_options = {
        "llama-3.3-70b-versatile": "🦙 Llama 3.3 70B (Best)",
        "llama-3.1-70b-versatile": "🦙 Llama 3.1 70B",
        "llama-3.1-8b-instant":   "⚡ Llama 3.1 8B (Fast)",
        "mixtral-8x7b-32768":     "🌪️ Mixtral 8x7B"
    }
    
    model_display = st.selectbox(
        "Llama model",
        options=list(model_options.values()),
        index=0,
    )
    model_choice = [k for k, v in model_options.items() if v == model_display][0]

    st.divider()
    st.subheader("🛠 Available Tools")
    for tool in TOOLS:
        with st.expander(tool.name):
            st.caption(tool.description)
    
    st.divider()
   
    if st.button("🗑 Clear conversation"):
        st.session_state.messages = []
        st.session_state.agent = None
        st.session_state.thread_id = str(uuid.uuid4())
        if "uploaded_file" in st.session_state:
            del st.session_state.uploaded_file
        if "file_content" in st.session_state:
            del st.session_state.file_content
        if "file_name" in st.session_state:
            del st.session_state.file_name
        st.rerun()
   
    st.caption(f"Model: `{model_choice}`")

# ── Session state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "file_content" not in st.session_state:
    st.session_state.file_content = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

if "agent" not in st.session_state or st.session_state.get("model") != model_choice:
    with st.spinner(f"Loading {model_choice}..."):
        st.session_state.agent = build_agent(model_choice)
        st.session_state.model = model_choice
        st.session_state.logger = SessionLogger()

# ── Header ──────────────────────────────────────────────────
st.title("🦙 Autonomous Llama Agent")
st.caption("Multi-step reasoning · Dynamic tool selection · Conversation memory")

# ── Chat history ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input Area with Attachment ─────────────────────────────
col_input, col_attach = st.columns([20, 1.5])

with col_input:
    user_input = st.chat_input("Ask anything... (e.g. Summarize the file)")

with col_attach:
    uploaded_file = st.file_uploader(
        label="",
        type=["pdf", "txt", "csv", "md", "docx"],
        key="chat_file_uploader",
        label_visibility="collapsed",
        help="Attach file"
    )

# Process uploaded file (using toast instead of permanent green message)
if uploaded_file is not None and st.session_state.uploaded_file != uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.file_content = uploaded_file.getvalue()
    st.session_state.file_name = uploaded_file.name
    st.toast(f"✅ File attached: **{uploaded_file.name}**", icon="📎")

# ── Process User Input ─────────────────────────────────────
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                agent_input = {
                    "messages": [HumanMessage(content=user_input)]
                }
                
                # Safely pass file to agent
                if st.session_state.file_content is not None:
                    agent_input["file_content"] = st.session_state.file_content
                    agent_input["file_name"] = st.session_state.file_name

                response = st.session_state.agent.invoke(agent_input, config=cfg)
                output = extract_output(response)
                
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})

            except Exception as e:
                st.error(f"Agent error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Error: {str(e)}"
                })

    st.rerun()
