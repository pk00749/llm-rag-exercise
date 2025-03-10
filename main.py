import warnings
import time
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever

warnings.filterwarnings("ignore")

model_path = 'pretrained_models/DeepSeek-R1-Distill-Qwen-7B-GGUF'
gguf_file_name = 'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf'

def init_existing_vector_database():
    embedding_function = SentenceTransformerEmbeddings(model_name="./pretrained_models/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma.db", embedding_function=embedding_function)
    return db

# def init_vectorstore_ollama(raw_texts):
#     raw_texts = text_splitter()
#     embeddings = OllamaEmbeddings(model="./pretrained_models/all-MiniLM-L6-v2")
#     ollama_db = Chroma.from_documents(persist_directory='./chroma.db', documents=raw_texts, embedding=embeddings)
#     return ollama_db

def init_pretrained_model():
    print("init model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=gguf_file_name)
    model = AutoModelForCausalLM.from_pretrained(model_path, gguf_file=gguf_file_name, load_in_4bit=True).half()
    return tokenizer, model

def init_retriever(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return retriever

def create_pipeline(tokenizer, model):
    pipe_query = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=1000, top_p=1,
                          truncation=True, repetition_penalty=1.15)
    llm = HuggingFacePipeline(pipeline=pipe_query)
    return llm

def create_chat_prompt_template():
    template = """Question: {question}

    Answer: Let's think step by step."""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt

def create_prompt_template():
    template = """引用: {context}
    问题: {question}
    答案: 我一步一步推理..."""
    prompt = PromptTemplate.from_template(template)
    return prompt

def create_chain(llm, question, context):
    prompt = create_prompt_template
    formatted_prompt = prompt.format(question=question, context=context)
    chain = formatted_prompt | llm
    return chain

def chat_v1():
    rag_time_before = time.time()
    chroma_db = init_existing_vector_database()
    rag_time_after = time.time()
    retrieval_time = rag_time_after - rag_time_before
    # ollama_db = init_vectorstore_ollama(texts)
    llm_time_before = time.time()
    tokenizer, model = init_pretrained_model()
    llm = create_pipeline(tokenizer, model)
    question = "谁参加了桃园三结义"
    related_docs = chroma_db.similarity_search(question, k=2)
    context = "\n".join([doc.page_content for doc in related_docs])
    prompt = create_prompt_template()
    formatted_prompt = prompt.format(question=question, context=context)
    response = llm.invoke(formatted_prompt)
    llm_time_after = time.time()
    llm_time = llm_time_after - llm_time_before
    total_time = retrieval_time + llm_time
    print(total_time)
    print(response)


# 3. 初始化检索器
# bm25_retriever = BM25Retriever.from_documents(texts)
# bm25_retriever.k = 10

# # 4. 初始化压缩检索器
# model = HuggingFaceCrossEncoder(model_name="/root/models/bce-reranker-base_v1")
# compressor = CrossEncoderReranker(model=model, top_n=5)
# ensemble_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.5, 0.5])
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=ensemble_retriever
# )

# # 6. 定义提示模板
# PROMPT_TEMPLATE_ANSWER = """
# 你是一个对话机器人，根据检索到的信息对用户的问题进行回答：
# context:{context}
# query:{query}
# """
# prompt_answer = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_ANSWER)
# chain_answer = prompt_answer | llm | StrOutputParser()

# 7. 格式化文档的辅助函数
# def format_docs(docs):
#     result = ""
#     for doc in docs:
#         if (len(result) + len(doc.page_content)) > 500:
#             break
#         result += doc.page_content
#     result = re.sub(r'[\n\r]+', ' ', result)
#     result = re.sub(r'([一二三四五六七八九十]+、|\d+\.)\s*', '', result)
#     return result

# 示例运行
if __name__ == "__main__":
    chat_v1()
    # query = "水浒传中宋江为什么落草为寇？"
    # retrieval_words, result, retrieval_time, llm_time, total_time = process(query)
    # print("回答结果：", result)
    # print("检索的语料：", retrieval_words)
    # print("检索时间：", retrieval_time)
    # print("大模型回答时间：", llm_time)
    # print("总时间：", total_time)
