from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


class NvidiaLLM:
    def __init__(self, model_name):
        self.llm = ChatNVIDIA(model=model_name)


def create_llm(model_name, model_type="NVIDIA"):
    # Use LLM to generate answer
    if model_type == "NVIDIA":
        model = NvidiaLLM(model_name)
    else:
        print("Error! Need model_name and model_type!")
        exit()

    return model.llm


class LLMClient:
    def __init__(self, model_name="mixtral_8x7b", model_type="NVIDIA"):
        self.llm = create_llm(model_name, model_type)

    def chat_with_prompt(self, system_prompt, prompt):
        langchain_prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )
        chain = langchain_prompt | self.llm | StrOutputParser()
        response = chain.stream({"input": prompt})

        return response

    def multimodal_invoke(
        self,
        b64_string,
        steer=False,
        creativity=0,
        quality=9,
        complexity=0,
        verbosity=8,
    ):
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image in detail:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_string}"},
                },
            ]
        )
        if steer:
            return self.llm.invoke(
                [message],
                labels={
                    "creativity": creativity,
                    "quality": quality,
                    "complexity": complexity,
                    "verbosity": verbosity,
                },
            )
        else:
            return self.llm.invoke([message])
