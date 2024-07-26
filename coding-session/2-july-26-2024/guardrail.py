from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import streamlit as st

llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1", nvidia_api_key=st.secrets["NVIDIA_API_KEY"])

def fact_check(evidence, query, response):

    system_message = f"""Your task is to conduct a thorough fact-check of a response provided by a large language model. You will be given the context documents as [[CONTEXT]], the original question posed by the user as [[QUESTION]], and the model's response as [[RESPONSE]]. Your primary objective is to meticulously verify each part of the model's response to ensure it aligns accurately and directly with the information presented in the context documents. Please refrain from using any external information or relying on prior knowledge. Focus on determining whether the response is entirely factual based on the provided context and whether it fully addresses the user's question. This process is crucial for maintaining the accuracy and reliability of the information given by the language model. You can provide suggestions to the user for follow-up questions based on the documents, that will provide them more information about the topic they are interested in. If your fact check returns True, start your reply with '**:green[TRUE]**' in your response, and if it returns False, start your reply with '**:red[FALSE]**' in your response."""
    
    user_message = f"""[[CONTEXT]]\n\n{evidence}\n\n[[QUESTION]]\n\n{query}\n\n[[RESPONSE]]\n\n{response}"""

    langchain_prompt = ChatPromptTemplate.from_messages([("system", system_message), ("user", "{input}")])

    chain = langchain_prompt | llm | StrOutputParser()
    response = chain.stream({"input": user_message})
    return response