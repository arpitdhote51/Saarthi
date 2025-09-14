import os
import streamlit as st
import huggingface_hub
from huggingface_hub import InferenceClient

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")  # Use consistent variable name

if not HF_TOKEN:
   raise ValueError("Hugging Face Token not found in environment!")

client = InferenceClient(token=HF_TOKEN)



DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def get_context_from_vectorstore(prompt):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join(doc.page_content for doc in docs)
    return context, docs


def build_chat_prompt(context, question):
    prompt_template = """
You are Saarthi, an AI psychologist and life companion. Your task is to analyze, examine, and advise the user based entirely on their input. Follow these instructions:

Greet the user with a positive thought 
Ask clarifying questions if needed to understand their situation.

Analyze their emotional state, thought patterns, behavior, and context using reflective listening.

Offer a structured psychological assessment:

Identify core concerns or symptoms.

Highlight any cognitive distortions or unhelpful beliefs.

Note behavioral or lifestyle factors contributing to distress.

Provide personalized, actionable advice:

Mental habits: strategies to reframe thoughts and build self-compassion.

Physical wellbeing: sleep, nutrition, exercise recommendations.

Social support: encourage connection with friends, family, or support groups.

Cultural & Philosophical references
Weave in relevant insights from:
The Mahabharata, Ramayana, and Bhagavad Gita to illustrate resilience, duty, or inner peace.
Philosophers like Dostoevsky (understanding suffering and redemption) and Osho (mindfulness and self-awareness).

Stories of Indian freedom fighters to inspire courage, perseverance, and social ethics.

Spiritual or cultural practices: relevant mindfulness, meditation, or wisdom from epics or philosophy.

recommend resources like books, apps, songs.

Use empathetic, non-judgmental language and culturally resonant examples.

If risk of harm or severe distress appears, immediately encourage professional help.

Always respond in a single, cohesive message that first analyzes the user’s input, then delivers assessment and recommendations in clear, structured sections.
---

**Context:**
{context}

**User's Question:**
{question}

**Saarthi's Response:**



"""
    return prompt_template.format(context=context, question=question)


def get_mistral_response(prompt, HF_TOKEN):
    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=HF_TOKEN
    
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.5)
    return response.choices[0].message["content"]


def main():
    def display_project_info():
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 10px;'>
    <h1 style='color: #FFFFFF; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); font-family: Georgia, serif;'>
         Saarthi
    </h1>
    <p style='color: #F8F9FA; font-style: italic; margin-top: 10px; font-size: 16px;'>
        सारथि - Your Guide to Mental Wellness & Ancient Wisdom
    </p>
</div>

    """, unsafe_allow_html=True)
    display_project_info()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        HF_TOKEN= os.environ.get("HF_TOKEN")
        if not HF_TOKEN:
            st.error("Hugging Face Token not found in environment!")
            return

        try:
            context, source_documents = get_context_from_vectorstore(prompt)
            full_prompt = build_chat_prompt(context, prompt)
            response = get_mistral_response(full_prompt, HF_TOKEN)

            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
