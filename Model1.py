import os
import logging
import json
import streamlit as st
from openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import torch
from streamlit_chat import message
import time
import tiktoken
from datetime import datetime
from streamlit_float import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app title
st.set_page_config(page_title="My Yoga Network Assistant", page_icon="🧘", layout="wide")
st.title("🧘 My Yoga Network Assistant")

# API Key and paths
DB_FAISS_PATH = 'vectorstores/db_faiss'
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "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")

custom_prompt_template = """You are a sophisticated and dedicated yoga assistance bot, programmed to provide accurate, respectful, and insightful responses. Your primary function is to offer information and insights related to yoga, focusing on its benefits in daily life and its impact on mental and physical health.

You are designed to adhere strictly to ethical guidelines, ensuring all your responses are free from harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. You maintain a socially unbiased stance and promote positivity and well-being in all interactions.

If you encounter a question that is unclear, nonsensical, or factually inconsistent, you are to clarify the confusion respectfully and guide the inquirer towards a coherent understanding, instead of providing incorrect or misleading information. In instances where you lack sufficient data or knowledge to respond accurately, you are to acknowledge the limitation openly, avoiding speculation or the dissemination of falsehoods.

Your ultimate aim is to educate, inform, and assist users in understanding yoga practices, their benefits, and the holistic impact of yoga on wellness, empowering them with reliable information to enhance their well-being.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Initialize Perplexity API client
client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

# Session state initialization
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful yoga assistant."}
    ]
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []

# Added state for new features
if 'uploaded_pic' not in st.session_state:
    st.session_state['uploaded_pic'] = False
if 'rerun' not in st.session_state:
    st.session_state['rerun'] = False

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def call_perplexity_api(full_prompt):
    """
    Function to call Perplexity API
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "engage in a helpful, detailed, polite conversation with a user."
            ),
        },
        {
            "role": "user",
            "content": full_prompt,
        },
    ]
    try:
        response = client.chat.completions.create(
            model="llama-3-sonar-large-32k-online",
            messages=messages,
        )
        message_content = response.choices[0].message.content
        usage = response.usage
        return message_content, usage
    except Exception as err:
        logger.error(f"Error calling Perplexity API: {err}")
        raise

def retrieval_qa_chain(prompt, db):
    """
    Retrieval QA Chain
    """
    def qa_chain(query):
        try:
            context = db.search(query, search_type="similarity", search_kwargs={'k': 2})
            full_prompt = prompt.format(context=context, question=query)
            answer, usage = call_perplexity_api(full_prompt)
            return {"result": answer, "usage": usage}
        except Exception as err:
            logger.error(f"Error in retrieval QA chain: {err}")
            raise
    return qa_chain

def qa_bot():
    """
    QA Model Function
    """
    try:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        qa_prompt = set_custom_prompt()
        qa = retrieval_qa_chain(qa_prompt, db)
        return qa
    except Exception as e:
        logger.error(f"Error initializing QA bot: {e}")
        raise

def generate_response(prompt):
    qa = qa_bot()
    response_data = qa(prompt)
    response = response_data["result"]
    usage = response_data["usage"]

    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.session_state['messages'].append({"role": "assistant", "content": response})
    st.session_state['generated'].append(response)
    st.session_state['past'].append(prompt)
    st.session_state['model_name'].append("llama-3-sonar-large-32k-online")
    st.session_state['total_tokens'].append(usage.total_tokens)
    st.session_state['cost'].append(calculate_cost(usage))

def calculate_cost(usage):
    total_tokens = usage.total_tokens
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000
    return cost

def display_conversation():
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")

def log_feedback(icon):
    st.toast("Thanks for your feedback!", icon="👌")
    last_messages = json.dumps(st.session_state["messages"][-2:])
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "
    activity += "positive" if icon == "👍" else "negative"
    activity += ": " + last_messages
    logger.info(activity)

@st.experimental_dialog("🎨 Upload a picture")
def upload_document():
    st.warning(
        "This is a demo dialog window. You need to process the file afterwards.",
        icon="💡",
    )
    picture = st.file_uploader(
        "Choose a file", type=["jpg", "png", "bmp"], label_visibility="hidden"
    )
    if picture:
        st.session_state["uploaded_pic"] = True
        st.rerun()

if "uploaded_pic" in st.session_state and st.session_state["uploaded_pic"]:
    st.toast("Picture uploaded!", icon="📥")
    del st.session_state["uploaded_pic"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "llama-3-sonar-large-32k-online"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_avatar = "👩‍💻"
assistant_avatar = "🤖"

if "rerun" in st.session_state and st.session_state["rerun"]:
    st.session_state["messages"].pop(-1)

for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)

if prompt or ("rerun" in st.session_state and st.session_state["rerun"]):
    with st.chat_message("assistant", avatar=assistant_avatar):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state["messages"]
            ],
            stream=True,
            max_tokens=300,
        )
        response = st.write_stream(stream)
    st.session_state["messages"].append({"role": "assistant", "content": response})
    if "rerun" in st.session_state and st.session_state["rerun"]:
        st.session_state["rerun"] = False

st.write("")

# If there is at least one message in the chat, we display the options
if len(st.session_state["messages"]) > 0:

    action_buttons_container = st.container()
    action_buttons_container.float(
        "bottom: 6.9rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
    )

    cols_dimensions = [7, 14.9, 14.5, 9.1, 9, 8.6, 8.7]
    cols_dimensions.append(100 - sum(cols_dimensions))
    col0, col1, col2, col3, col4, col5, col6, col7 = action_buttons_container.columns(
        cols_dimensions
    )

    with col1:
        json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")
        st.download_button(
            label="📥 Save!",
            data=json_messages,
            file_name="chat_conversation.json",
            mime="application/json",
        )

    with col2:
        if st.button("Clear 🧹"):
            st.session_state["messages"] = []
            if "uploaded_pic" in st.session_state:
                del st.session_state["uploaded_pic"]
            st.rerun()

    with col3:
        if st.button("🎨"):
            upload_document()

    with col4:
        if st.button("🔁"):
            st.session_state["rerun"] = True
            st.rerun()

    with col5:
        if st.button("👍"):
            log_feedback("👍")

    with col6:
        if st.button("👎"):
            log_feedback("👎")

    with col7:
        enc = tiktoken.get_encoding("cl100k_base")
        tokenized_full_text = enc.encode(
            " ".join([item["content"] for item in st.session_state["messages"]])
        )
        label = f"💬 {len(tokenized_full_text)} tokens"
        st.link_button(label, "https://platform.openai.com/tokenizer")

else:
    if "disclaimer" not in st.session_state:
        with st.empty():
            for seconds in range(3):
                st.warning(
                    "‎ You can click on 👍 or 👎 to provide feedback regarding the quality of responses.",
                    icon="💡",
                )
                time.sleep(1)
            st.write("")
            st.session_state["disclaimer"] = True

def main():
    """
    Streamlit UI
    """
    st.sidebar.header("Yoga Network Assistant")
    st.sidebar.markdown("Navigate through the options to explore more about yoga and its benefits.")

    st.sidebar.header("Navigation")
    st.sidebar.button("Home")
    st.sidebar.button("Discover")
    st.sidebar.button("Library")

    st.sidebar.header("Contact")
    st.sidebar.info("For more information, visit [My Yoga Network](https://myyoganetwork.com)")

if __name__ == '__main__':
    main()