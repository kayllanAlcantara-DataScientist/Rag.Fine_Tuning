# build_vector_db.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Carregar os documentos
loader = DirectoryLoader('./knowledge_base/', glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
documents = loader.load()
print(f"Carregados {len(documents)} documentos.")

# 2. Dividir em chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"Documentos divididos em {len(texts)} chunks.")

# 3. Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# 4. Chroma
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./db"
)
vector_store.persist()
print("Banco vetorial salvo em './db'.")

# Teste rápido
query = "cansaço e perda de interesse"
retrieved = vector_store.similarity_search(query, k=1)
if retrieved:
    print("Encontrado:", retrieved[0].page_content[:400])
else:
    print("Nenhum documento retornado.")
