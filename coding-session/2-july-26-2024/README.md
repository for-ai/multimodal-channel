# Multimodal RAG with NVIDIA NIM and Milvus VectorDB

This tutorial will walkthrough you on how to create your own Multimodal RAG version that you can talk with your image using NVIDIA NIM and Milvus Vector Database

## Setup

__Preliminary__:

- Make sure that your Docker is on and ready to run
- If you do not have NVIDIA NIM API KEY yet, you can start creating at here: [NVIDIA NIM Try Out!](https://build.nvidia.com/explore/discover)
    - Your API Key should have something similar like this: `nvapi-xxx`

1. Create your virtual environment with `venv` or `conda`. Prefer `python=3.11`
2. Activate your virtual environment and run `pip install -r requirements.txt`
3. Run `export NVIDIA_API_KEY='nvapi-xxx'`
4. Start streamlit `streamlit run app.py`