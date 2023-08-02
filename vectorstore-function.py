import os 
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DeepLake

load_dotenv()

#Load the document
document_name = 'course-materials.pdf'
raw_documents = PyPDFLoader(document_name)
documents = raw_documents.load_and_split()
embeddings = OpenAIEmbeddings()


#Create Vectorstore
db = DeepLake.from_documents(documents, dataset_path="<YOUR_DEEPLAKE_DATASET_PATH>", embedding=embeddings)