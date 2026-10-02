import os
import streamlit as st
import random
from datasets import load_dataset
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="Pediatric Medical Chatbot",
    page_icon="👶",
    layout="centered"
)

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

PEDIATRIC_KEYWORDS = [
    "child", "children", "kid", "kids", "baby", "babies",
    "infant", "infants", "newborn", "newborns", "neonatal",
    "pediatric", "pediatrics", "toddler", "toddlers",
    "my son", "my daughter", "my child", "my baby", "my kid",
    "infancy", "adolescent", "teenager"
]
NUM_SAMPLES = 5000
SIMILARITY_THRESHOLD = 0.85
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def build_vectorstore():
    with st.spinner("📥 Loading medical dataset... (first time only)"):
        dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

        def is_pediatric(example):
            text = (example["Description"] + " " + example["Patient"]).lower()
            return any(kw in text for kw in PEDIATRIC_KEYWORDS)

        pediatric_data = dataset.filter(is_pediatric)

    with st.spinner("🧹 Preparing pediatric Q&A pairs..."):
        random.seed(42)
        indices = random.sample(range(len(pediatric_data)), min(NUM_SAMPLES, len(pediatric_data)))

        qa_data = []
        for i in tqdm(indices):
            row = pediatric_data[i]
            question = row["Patient"].strip()
            answer = row["Doctor"].strip()
            if len(question) > 20 and len(answer) > 50:
                qa_data.append({"question": question, "answer": answer})

    with st.spinner("🧠 Loading embedding model..."):
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    with st.spinner("🗄️ Building vector database... (2-4 min, first time only)"):
        documents = [
            Document(page_content=item["question"], metadata={"answer": item["answer"]})
            for item in qa_data
        ]

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name="pediatric_qa"
        )

    return vectorstore


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.3,
    )


def extract_text(response):
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def search_web(query, max_results=3):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None
        return "\n\n".join([
            f"Source: {r.get('title', '')}\n{r.get('body', '')}"
            for r in results
        ])
    except Exception as e:
        print(f"Web search error: {e}")
        return None


def get_bot_response(user_question, vectorstore, llm):
    results = vectorstore.similarity_search_with_score(user_question, k=5)
    good_results = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    context_parts = []
    source_used = ""

    if good_results:
        source_used = "📚 Medical Database (Pediatric)"
        for i, (doc, score) in enumerate(good_results[:4], 1):
            context_parts.append(
                f"[Case {i}]\nQuestion: {doc.page_content}\nAnswer: {doc.metadata['answer']}"
            )
    else:
        source_used = "🌐 Trusted Medical Sources (Web)"
        web_results = search_web(f"pediatric {user_question}", max_results=3)
        context_parts.append(web_results if web_results else "No external source found.")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a friendly and helpful pediatric medical assistant chatbot.

YOUR TASK:
Use the medical cases provided in the CONTEXT below to give a helpful, informative answer to the parent's question.

GUIDELINES:
1. Extract the most relevant medical information from the context and present it clearly.
2. Be helpful and informative — provide practical advice (home care, what to watch for, when to see a doctor).
3. Use simple, warm language that a parent can understand.
4. Structure your answer with clear points (use bullet points when helpful).
5. If the context has partial information, use what's available and mention that consulting a pediatrician is recommended.
6. Only if the context is completely unrelated or empty, say you don't have specific info.
7. Do NOT prescribe specific medication doses.
8. Always end with: "⚕️ This information is for educational purposes only. If symptoms worsen or persist, please consult a licensed pediatrician."

CONTEXT:
{context}

PARENT'S QUESTION:
{user_question}

HELPFUL ANSWER:"""

    try:
        response = llm.invoke(prompt)
        answer = extract_text(response)
    except Exception as e:
        answer = f"Sorry, an error occurred: {e}"

    return answer, source_used


st.title("👶 Pediatric Medical Chatbot")
st.markdown("### Ask any question about your child's health")
st.markdown("⚕️ **Disclaimer:** This chatbot is for educational purposes only. Always consult a licensed pediatrician.")

with st.spinner("🚀 Starting up... (first run may take 5-10 minutes)"):
    vectorstore = build_vectorstore()
    llm = get_llm()

st.success("✅ Chatbot is ready! Ask your question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about your child's health..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, source = get_bot_response(user_input, vectorstore, llm)
            full_response = f"**{source}**\n\n{answer}"
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
