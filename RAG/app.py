import os
import streamlit as st
from dotenv import load_dotenv

# LangGraph + LangChain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from config import config

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -------------------------------------------------
# Initialize Embeddings, Vector DB, LLM
# -------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=config.EMBEDDING_MODEL_NAME
)

vectordb = Chroma(
    persist_directory=config.CHROMA_PERSIST_DIRECTORY,
    embedding_function=embeddings
)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_tokens=400
)

# -------------------------------------------------
# LangGraph Node
# -------------------------------------------------

def call_model(state: MessagesState):
    system_prompt = (
        "You are a PDF-based academic assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not present in the documents, say 'I don't know'. "
        "Keep answers concise (max 3 sentences)."
    )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": response}

def build_graph():
    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")
    return graph.compile(checkpointer=MemorySaver())

app = build_graph()

# -------------------------------------------------
# 🧠 MEMORY HELPERS
# -------------------------------------------------

def resolve_followup_question(prompt: str):
    """
    Handles vague follow-up questions like:
    - what are those types
    - repeat my last question
    """
    vague_phrases = [
        "what are those types",
        "those types",
        "repeat my last question",
        "repeat last question",
        "explain again"
    ]

    if prompt.lower() in vague_phrases:
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user" and msg["content"].lower() not in vague_phrases:
                return msg["content"]

    return prompt


def resolve_pronoun_question(prompt: str):
    """
    Handles pronouns like:
    - this
    - that
    - it
    - those
    """
    pronouns = ["this", "that", "it", "those", "them"]

    if any(p in prompt.lower() for p in pronouns):
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                # Extract topic from assistant answer (first sentence)
                topic = msg["content"].split(".")[0]
                return f"Tell me more about {topic}"

    return prompt

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.set_page_config(page_title="📄 PDF Academic QA", layout="centered")
st.title("📄 PDF Academic Question Answering")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "pdf_chat_session"

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------------------------
# Chat Logic
# -------------------------------------------------

if user_input := st.chat_input("Ask a question from the PDFs..."):

    # 🧠 Resolve memory-based questions
    resolved_question = resolve_followup_question(user_input)
    resolved_question = resolve_pronoun_question(resolved_question)

    # Store resolved user question
    st.session_state.messages.append({
        "role": "user",
        "content": resolved_question
    })

    with st.chat_message("user"):
        st.markdown(resolved_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):

            # 🔍 Retrieval
            docs = vectordb.similarity_search_with_score(
                resolved_question,
                k=3
            )

            context = "\n\n".join([doc.page_content for doc, _ in docs])

            message = HumanMessage(
                content=f"""
Context from documents:
{context}

Question:
{resolved_question}
"""
            )

            result = app.invoke(
                {"messages": [message]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )

            answer = result["messages"][-1].content

            # 📌 Sources
            source = docs[0][0].metadata.get("source", "N/A") if docs else "N/A"
            pages = sorted(
                set(str(d[0].metadata.get("page", "N/A")) for d in docs)
            )

            final_answer = (
                f"{answer}\n\n"
                f"**Source:** {source}\n"
                f"**Pages:** {', '.join(pages)}"
            )

            st.markdown(final_answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer
            })

# -------------------------------------------------
# Run:
# streamlit run app.py
# -------------------------------------------------
