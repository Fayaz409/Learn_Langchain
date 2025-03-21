from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
load_dotenv()

loader = TextLoader(r"C:\Users\ad\Documents\LearnAgents\data_loader\data.txt")
text = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separators=['.','!','?'],
    length_function=len
)

model = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),

)
embedding = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv('GEMINI_API_KEY'), model="models/text-embedding-004")

texts = splitter.split_documents(text)
vector_db = FAISS.from_documents(texts,embedding)

retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3})


template = """
Only give the answer if it the context has the data about the questions topic.
otherwise say "I don't know"
Context: {context}
Question:{question}
give the answer in following language:{language}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context":itemgetter("question")|retriever,
     "question":itemgetter("question"),
     "language":itemgetter("language")}
    |prompt
    |model
    |StrOutputParser()
)
print("English")
print(chain.invoke({"question":"Who is Kaelin","language":"english"}))
print("Spanish")
print(chain.invoke({"question":"Who is Kaelin","language":"spanish"}))
print("French")
print(chain.invoke({"question":"Who is Kaelin","language":"french"}))