import os
import warnings
import time
from llama_cpp import Llama
from langchain.prompts import PromptTemplate
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
os.environ["LLAMA_LOG_LEVEL"] = "none"

model_path = 'pretrained_models/DeepSeek-R1-Distill-Qwen-14B-GGUF'
gguf_file_name = 'DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf'

template = """
根据以下上下文来回答问题。如果不知道答案，就直接说不知道，不能试图编造答案。
问题: {question}

上下文: {context}

答案: 
"""


def query_vector_database(question):
    print("Querying vector database...")
    client = MilvusClient("./milvus_demo.db")
    # 加载嵌入模型
    model = SentenceTransformer("./pretrained_models/bge-large-zh-v1.5")
    embed_query = model.encode(question)
    res = client.search(collection_name="demo_collection",
                        data=[embed_query],
                        limit=3,
                        output_fields=["text"])
    retrieved_text_lists = []
    for hints in res:
        for hint in hints:
            hint.get('entity')
            retrieved_text_lists.append(hint.get('entity').get('text'))

    context = "\n".join(retrieved_text_lists)
    print("Done")
    return context


def create_chat(question, context):
    print(f"Initialing model {gguf_file_name}...")
    llm = Llama(model_path=f"./{model_path}/{gguf_file_name}", n_gpu_layers=-1, n_threads=10, n_threads_batch=10)
    print("Done")
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(question=question, context=context)
    print(formatted_prompt)
    print("推理中...")
    output = llm(prompt=formatted_prompt, max_tokens=500)
    print(output['choices'][0]['text'])

def chat(question):
    # rag
    rag_time_before = time.time()
    context = query_vector_database(question=question)
    rag_time_after = time.time()
    rag_time = rag_time_after - rag_time_before

    # llm
    llm_time_before = time.time()
    create_chat(question=question, context=context)
    llm_time_after = time.time()
    llm_time = llm_time_after - llm_time_before
    print(f"rag time:{rag_time}")
    print(f"llm time: {llm_time}")


# 示例运行
if __name__ == "__main__":
    chat("夏侯渊最后战死了吗")
