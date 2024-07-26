import os
import base64
import argparse
import pandas as pd
from PIL import Image
from io import BytesIO

import streamlit as st
import streamlit_analytics

from config import get_config
from memory import init_memory, get_summary, add_history_to_memory
from llm import LLMClient
from embedder import NVIDIAEmbedders
from vectordb import MilvusVectorClient
from retriever import Retriever

LLM_CLIENT = LLMClient("mixtral_8x7b")

##########################################################
# Start the analytics service (using browser.usageStats) # 
##########################################################
streamlit_analytics.start_tracking()

st.set_page_config(
    page_title="Multimodal RAG Assistant",
    page_icon=":speech_balloon:",
    layout="wide",
)

@st.cache_data()
def load_config(cfg_arg):
    try:
        config = get_config(os.path.join(cfg_arg + ".config"))
        return config
    except Exception as e:
        print("Error loading config:", e)
        return None

################################################################
# Initialize session state variables if not already present    #
################################################################
if "prompt_value" not in st.session_state:
    st.session_state["prompt_value"] = None

if "config" not in st.session_state:
    st.session_state.config = load_config("multimodal")
    print(st.session_state.config)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Ask me a question!"}]
if "sources" not in st.session_state:
    st.session_state.sources = []

if "image_query" not in st.session_state:
    st.session_state.image_query = ""

if "queried" not in st.session_state:
    st.session_state.queried = False

if "memory" not in st.session_state:
    st.session_state.memory = init_memory(
        LLM_CLIENT.llm, st.session_state.config["summary_prompt"]
    )

MEMORY = st.session_state.memory

####################################################################
# BOT CONFIGURATION                                                #
####################################################################

with st.sidebar:
    prev_cfg = st.session_state.config
    try:
        defaultidx = [["multimodal"]].index(st.session_state.config["name"].lower())
    except Exception:
        defaultidx = 0
    st.header("Bot Configuration")
    cfg_name = st.selectbox(
        "Select a configuration/type of bot.", (["multimodal"]), index=defaultidx
    )
    st.session_state.config = get_config(
        os.path.join(cfg_name + ".config")
    )
    config = get_config(os.path.join(cfg_name + ".config"))
    if st.session_state.config != prev_cfg:
        st.experimental_rerun()

    st.success("Select an experience above.")

    st.header("Image Input Query")

    # with st.form("my-form", clear_on_submit=True):
    uploaded_file = st.file_uploader(
        "Upload an image (JPG/JPEG/PNG) along with a text input:",
        accept_multiple_files=False,
    )
    #    submitted = st.form_submit_button("UPLOAD!")

    if uploaded_file and st.session_state.image_query == "":
        st.success("Image loaded for multimodal RAG Q&A.")
        st.session_state.image_query = os.path.join("tmp", uploaded_file.name)
        with open(st.session_state.image_query, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Getting image description using NeVA"):
            neva = LLMClient("nvidia/neva-22b")
            image = Image.open(st.session_state.image_query).convert("RGB")
            buffered = BytesIO()
            image.save(
                buffered, format="JPEG", quality=20
            )  # Quality = 20 is a workaround (WAR)
            b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            res = neva.multimodal_invoke(
                b64_string, creativity=0, quality=9, complexity=0, verbosity=9
            )
            st.session_state.image_query = res.content

    if not uploaded_file:
        st.session_state.image_query = ""

# Page title
st.header(config["page_title"])
st.markdown(config["instructions"])

######################################################
# VECTOR DATABASE INITIALIZE                         #
######################################################
if (
    "vector_client" not in st.session_state
    or st.session_state.vector_client.collection_name
    != config["core_docs_directory_name"]
):
    try:
        st.session_state.vector_client = MilvusVectorClient(
            hostname="localhost",
            port="19530",
            collection_name=config["core_docs_directory_name"],
        )
    except Exception as e:
        st.write(
            f"Failed to connect to Milvus vector DB, exception: {e}. Please follow steps to initialize the vector DB, or upload documents to the knowledge base and add them to the vector DB."
        )
        st.stop()

######################################################
# EMBEDDING MODEL INITIALIZE                         #
######################################################
if "query_embedder" not in st.session_state:
    st.session_state.query_embedder = NVIDIAEmbedders(
        name="NV-Embed-QA", type="query"
    )

######################################################
# RETRIEVAL INITIALIZE                               #
######################################################
if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever(
        embedder=st.session_state.query_embedder,
        vector_client=st.session_state.vector_client,
    )
retriever = st.session_state.retriever

