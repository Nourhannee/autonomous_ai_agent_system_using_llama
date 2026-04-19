"""
streamlit_app.py - Web UI for the Autonomous Llama Agent.
Run: streamlit run streamlit_app.py
"""

import streamlit as st
from agent import build_agent, TOOLS, extract_output
from logger import SessionLogger
import config, time, uuid

st.set_page_config(
    page_title="Autonomous Llama Agent",
    page_icon="🦙",
    layout="wide",
)

st.markdown("""
<style>
.agent-action  { background:#fff8ec; border-left:3px solid #EF9F27; padding:8px 12px; border-radius:0 6px 6px 0; font-size:13px; margin:4px 0; }
.agent-obs     { background:#f0faf5; border-left:3px solid #1D9E75; padding:8px 12px; border-radius:0 6px 6px 0; font-size:13px; margin:4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    model_choice = st.selectbox(
        "Llama model",
        ["llama3.2:3b", "llama3.1", "llama3.1:70b", "mistral"],
        index=0,
    )
    st.divider()
    st.subheader("🛠 Available Tools")
    for tool in TOOLS:
        with st.expander(tool.name):
            st.caption(tool.description)
    st.divider()
    if st.button("🗑 Clear conversation"):
        st.session_state.messages   = []
        st.session_state.agent      = None
        st.session_state.thread_id  = str(uuid.uuid4())
        st.rerun()
    st.caption(f"Model: `{model_choice}`")

# ── Session state ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "agent" not in st.session_state or st.session_state.get("model") != model_choice:
    with st.spinner(f"Loading {model_choice}..."):
        st.session_state.agent  = build_agent(model_choice)
        st.session_state.model  = model_choice
        st.session_state.logger = SessionLogger()

# ── Header ──────────────────────────────────────────────────
st.title("🦙 Autonomous Llama Agent")
st.caption("Multi-step reasoning · Dynamic tool selection · Conversation memory")

# ── Chat history ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ───────────────────────────────────────────────────
user_input = st.chat_input("Ask anything...  e.g. 'What is the weather in Cairo?'")

if user_input:
    from langchain_core.messages import HumanMessage

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            t0 = time.time()
            try:
                # ✅ pass thread_id in config so MemorySaver works
                cfg = {"configurable": {"thread_id": st.session_state.thread_id}}

                response = st.session_state.agent.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=cfg,
                )
                elapsed = time.time() - t0
                output  = extract_output(response)

                st.markdown(output)
                col1, col2 = st.columns(2)
                col1.metric("Time", f"{elapsed:.1f}s")

                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": output,
                })
                st.session_state.logger.log(
                    user_input,
                    {"output": output, "intermediate_steps": []}
                )

            except Exception as e:
                st.error(f"Agent error: {e}")
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": f"Error: {e}",
                })