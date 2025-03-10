from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv() 
file = os.path.dirname(__file__)
file_path = os.path.join(file, "data.txt")
loader = TextLoader(file_path)
text = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=['.','!','?'],
    length_function=len
)

texts = splitter.split_documents(text)
# print('Texts Recursive Character Text Splitter',texts)
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv('GEMINI_API_KEY'), model="models/text-embedding-004")

vector_db = FAISS.from_documents(texts, embedding_model)
results = vector_db.similarity_search(
    "The Legacy of a Shepherd",
    k=2,
)

print('Results\n',results)
# embeddings_ = embeddings.embed_documents([doc.page_content for doc in texts])
# print(embeddings_)

# print(len(embeddings_[0]))