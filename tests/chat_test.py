import multiprocessing
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from api.config import MODELS_PATH
from api.schemas.chat import *
from api.services.chat import *

model_kwargs = {
    "n_gpu_layers": -1,
    "n_batch": 512,
    "n_ctx": 1024,
    "temperature": 0.8,
    "repeat_penalty": 1.2,
    "top_k": 20,
    "top_p": 0.9,
    "n_threads": multiprocessing.cpu_count() - 1,
    "f16_kv": True,
    "vocab_only": False,
    "use_mlock": False,
    "max_tokens": 1024,
    "verbose": False,
    "streaming": True,
}

model = "Meta-Llama-3-8B-Instruct-Q8_0.gguf"
messages = [
    Message(role="system", content="You are a chat assistant, always answer briefly and informatively, use the chat history to generate the responses."),
    Message(role="user", content="My name is John"),
    Message(role="assistant", content="Hello John, how can I help you today?"),
    Message(role="user", content="I need you to tell me my name"),
]

chat_messages = format_chat_message(messages)

chat_prompt = ChatPromptTemplate.from_messages(chat_messages)

parser = StrOutputParser()

llm = load_chat_llm(model_name=f"{MODELS_PATH}{model}", **model_kwargs)

chain = chat_prompt | llm | parser

res = chain.invoke({})

print(res)
# response = generate_reponse(chat_messages, llm)

# print(response.content)