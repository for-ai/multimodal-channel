from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_core.prompts.prompt import PromptTemplate

def init_memory(llm, prompt_str):
    SUMMARY_PROMPT = PromptTemplate(
            input_variables=["summary", "new_lines"], template=prompt_str
    )
    memory = ConversationSummaryMemory(llm=llm, prompt=SUMMARY_PROMPT)

    return memory

def get_summary(memory):
    return memory.buffer

def add_history_to_memory(memory, input_str, output_str):

    # add message to memory
    chat_memory = memory.chat_memory
    chat_memory.add_user_message(input_str)
    chat_memory.add_ai_message(output_str)

    # generate new summary
    buffer = memory.buffer
    new_buffer = memory.predict_new_summary(
            chat_memory.messages[-2:], buffer
            )
    # update buffer
    memory.buffer = new_buffer
    print("\n\nUpdated summary: ", new_buffer)

    return memory