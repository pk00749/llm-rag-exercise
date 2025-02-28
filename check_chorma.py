import chromadb

client = chromadb.PersistentClient(path="./chroma.db")
# 获取所有集合的列表
collections = client.list_collections()

# 打印集合名称
for collection in collections:
    print(collection)