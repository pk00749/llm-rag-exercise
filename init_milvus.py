from pymilvus import MilvusClient, DataType, model
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import DirectoryLoader, TextLoader

def text_splitter():
    print("Splitting text...")

    with open(r'三国演义.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # for multi docs
    # data = DirectoryLoader("./", glob=r"*.txt", loader_cls=TextLoader).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", " ", "。", "！", "？", "；", "...", ".", "!", "?", ";"]
    )
    chunks = splitter.split_text(text)
    print("Done.")
    return chunks


def init_milvus():
    print("Initialling Milvus...")
    client = MilvusClient("./milvus_demo.db")

    # create schema and collection
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=512)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535),
    schema.add_field(field_name="metadata", datatype=DataType.VARCHAR, max_length=65535)

    client.create_collection(
        collection_name="demo_collection",
        dimension=512,  # The vectors we will use in this demo has 384 dimensions
        schema = schema
    )

    # create index
    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        metric_type= "L2",
        index_type= "IVF_FLAT",
        params= {"nlist": 128}
    )
    client.create_index(collection_name="demo_collection", index_params=index_params)
    print("Done")
    return client


def insert_vector_data(client, chunks):
    client = MilvusClient("./milvus_demo.db")
    print("Inserting vectors into Milvus...")
    # 加载嵌入模型
    model = SentenceTransformer("./pretrained_models/bge-large-zh-v1.5")
    vectors = model.encode(chunks)
    print(vectors.shape)

    # 插入数据
    length_vectors = len(vectors)
    # mr = collection.insert([[vectors[i].tolist(), split_text[i], "metadata"] for i in range(length_spilt_text)])
    # collection.flush()  # 确保数据写入

    data = [{"id": i, "vector": vectors[i], "text": chunks[i], "metadata": "sanguo"} for i in range(length_vectors)]
    res = client.insert(
        collection_name="demo_collection",
        data=data
    )
    print("Done")

    query = "谁参与桃园结义"
    embed_query = model.encode(query)
    res = client.search(collection_name="demo_collection",
                                     data=[embed_query],
                                     limit=1)
    for hints in res:
        for hint in hints:
            print(hint)


def query_vector_data(query):
    print("querying...")
    client = MilvusClient("./milvus_demo.db")
    print("Inserting vectors into Milvus...")
    # 加载嵌入模型
    model = SentenceTransformer("./pretrained_models/bge-large-zh-v1.5")
    embed_query = model.encode(query)
    res = client.search(collection_name="demo_collection",
                         data=[embed_query],
                         limit=1,
                        output_fields=["text"])
    # retrieved_texts = [result[0].entity.get("text") for result in res]
    retrieved_text_lists = []
    for hints in res:
        for hint in hints:
            print(hint)
            hint.get('entity')
            retrieved_text_lists.append(hint.get('entity').get('text'))
    print("Done")
    return retrieved_text_lists

if __name__ == "__main__":
    # text_chunks = text_splitter()
    # client = init_milvus()
    # insert_vector_data(client, text_chunks)
    query_vector_data("谁参与桃园结义")