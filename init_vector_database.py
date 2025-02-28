import time
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

def text_splitter():
    print("Splitting text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", "。", "！", "？", "；", "...", ".", "!", "?", ";"]
    )
    data = DirectoryLoader("./", glob=r"*.txt", loader_cls=TextLoader).load()
    texts = splitter.split_documents(data)

    return texts

# def token_text_splitter():
#     splitter = TokenTextSplitter(
#         chunk_size=100,
#         chunk_overlap=50
#     )
#     data = DirectoryLoader("./", glob=r"*.txt", loader_cls=TextLoader).load()
#     texts = splitter.split_documents(data)
#
#     return texts


def init_vector_database(raw_texts):
    print("Initialing vector database...")
    embedding_function = SentenceTransformerEmbeddings(model_name="./pretrained_models/all-MiniLM-L6-v2")
    db = Chroma.from_documents(persist_directory='./chroma.db', documents=raw_texts, embedding=embedding_function)
    return db


if __name__ == "__main__":
    rag_time_before = time.time()
    texts = text_splitter()
    chroma_db = init_vector_database(texts)
    rag_time_after = time.time()