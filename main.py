import streamlit as st
from dataclasses import dataclass
from typing import Literal

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Message dataclass ---
@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# --- CSS ---
def load_css():
    st.markdown("""
    <style>
    .chat-row { display: flex; margin: 5px; width: 100%; }
    .row-reverse { flex-direction: row-reverse; }
    .chat-bubble {
        font-family: "Source Sans Pro", sans-serif;
        padding: 8px 12px;
        margin: 4px 10px;
        max-width: 70%;
    }
    .ai-bubble {
        background: #f1f4f8;
        color: #333;
        border-radius: 12px;
    }
    .human-bubble {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #003049;
        border-radius: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Init state ---
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "user-session"
    if "greeted" not in st.session_state:
        st.session_state.greeted = False

# --- Sidebar: intro + user form ---
with st.sidebar:
    st.markdown("## 🧠 MindEase")
    st.markdown("""
**MindEase** is a next-gen AI-powered emotional wellness companion.

Built with:
- 🧠 **Gemini 1.5 Flash** via Google Generative AI
- 🛠️ **LangChain** memory + chat architecture
- ⚡ **Streamlit** for interactive UI
- 📦 Modular, session-aware design

### What it offers:
- Emotionally intelligent responses
- Context memory during conversation
- Personalized support based on your mood
- A safe, judgment-free space to breathe and reflect

This is more than a chatbot — it’s a calming AI therapist powered by cutting-edge tech.
""")

    st.markdown("### ✨ Tell me about you")

    st.text_input("Your first name", key="user_name")
    st.selectbox(
        "How are you feeling today?",
        ["", "Stressed", "Overwhelmed", "Okay", "Curious", "Exhausted"],
        index=0,
        key="user_feeling"
    )

# --- Store history per session ---
store = {}
def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# --- LLM setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=st.secrets["gemini_api_key"],
    temperature=0.6
)

# --- Summarize last user messages ---
def summarize_conversation():
    user_msgs = [msg.message for msg in st.session_state.history if msg.origin == "human"]
    if not user_msgs:
        return ""
    recent = "\n".join(user_msgs[-3:])
    return f"The user recently shared:\n{recent}\n"

# --- Prompt builder ---
def build_prompt():
    memory_summary = summarize_conversation()
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are MindEase, a calm, present, and emotionally intelligent AI therapist. "
         "You gently help users reflect, reduce stress, and feel heard. "
         "Avoid repeating the same advice. Respond with empathy and presence.\n"
         + memory_summary.strip()
        ),
        MessagesPlaceholder(variable_name="messages")
    ])

def get_chain():
    return RunnableWithMessageHistory(
        build_prompt() | llm,
        get_session_history=get_history,
        input_messages_key="messages",
        history_messages_key="messages"
    )

# --- On user submission ---
def on_click():
    user_input = st.session_state.human_prompt
    memory = get_history(st.session_state.session_id)

    # 1. Save user message in UI + memory
    st.session_state.history.append(Message("human", user_input))
    memory.add_user_message(user_input)

    # 2. Inject personalized AI intro (only once)
    if not st.session_state.greeted:
        name = st.session_state.get("user_name", "friend")
        feeling = st.session_state.get("user_feeling", "").lower()

        intro = f"Hello {name.capitalize()}, I’m MindEase 🧘 – your gentle relaxation companion."
        if feeling:
            intro += f" I see you're feeling {feeling} today. Thank you for being here."
        intro += " Let's talk through what's on your mind."

        memory.add_ai_message(intro)
        st.session_state.history.append(Message("ai", intro))
        st.session_state.greeted = True
        st.session_state.human_prompt = ""  # clear field
        return  # ⛔ Skip model call on first message

    # 3. For ongoing messages, use the chain
    chain_with_history = get_chain()
    response = chain_with_history.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"session_id": st.session_state.session_id}}
    )

    st.session_state.history.append(Message("ai", response.content))
    st.session_state.human_prompt = ""  # ✅ clear input

# --- App startup ---
init_state()
load_css()

st.title("🧘 MindEaseAI")
st.markdown("Let’s process your thoughts gently, with calm presence and emotional memory.")

chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

# --- Chat rendering ---
with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row {'row-reverse' if chat.origin == 'human' else ''}">
            <div class="chat-bubble {'human-bubble' if chat.origin == 'human' else 'ai-bubble'}">
                {chat.message}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

# --- Input UI ---
with prompt_placeholder:
    st.markdown("**What’s on your mind?**")
    cols = st.columns((6, 1))
    cols[0].text_input("Your message", key="human_prompt", label_visibility="collapsed")
    cols[1].form_submit_button("Submit", type="primary", on_click=on_click)
