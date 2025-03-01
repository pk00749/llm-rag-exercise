import warnings
import time
from llama_cpp import Llama
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore")

model_path = 'pretrained_models/DeepSeek-R1-Distill-Qwen-14B-GGUF'
gguf_file_name = 'DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf'

template = """
根据以下上下文来回答问题。如果不知道答案，就直接说不知道，不能试图编造答案。
上下文: {context}

问题: {question}

答案: 
"""


def init_existing_vector_database(question):
    embedding_function = HuggingFaceBgeEmbeddings(model_name="pretrained_models/bge-large-zh")
    db = Chroma(persist_directory="./chroma.db", embedding_function=embedding_function)
    related_docs = db.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in related_docs])
    return context

def init_pretrained_model():
    print(f"init model {gguf_file_name}...")
    llm = Llama(model_path=f"./{model_path}/{gguf_file_name}")
    return llm

def create_chat(llm, question, context):
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(question=question, context=context)
    print(formatted_prompt)
    output = llm(prompt=formatted_prompt, max_tokens=1500)
    print(output['choices'][0]['text'])

def chat(question):
    llm_time_before = time.time()
    llm = init_pretrained_model()
    context = init_existing_vector_database(question=question)
    create_chat(llm=llm, question=question, context=context)
    llm_time_after = time.time()
    llm_time = llm_time_after - llm_time_before
    print(llm_time)


# 示例运行
if __name__ == "__main__":
    chat("刘备最后统一三国")
