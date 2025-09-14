#load raw data/pdf
# create chunks
# create embeddings
# store in vector database 
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_PATH = 'data/'
def load_pdf_files(data):
    # Load the PDF files from the directory
    loader = DirectoryLoader(data,glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
#print(f"Number of documents: {len(documents)}")

#create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_documents(extracted_data)
    return chunks

chunks = create_chunks(extracted_data=documents)
#print(f"Number of chunks: {len(chunks)}")

#create embeddings
def get_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embedding_model
embedding_model = get_embeddings()


#store in vector database
DB_FAISS_PATH='vectorstore/db_faiss'
db=FAISS.from_documents(chunks,embedding_model)
db.save_local(DB_FAISS_PATH)