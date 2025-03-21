from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
# from langchain_community
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv() 
# file = os.path.dirname(r"C:\Users\ad\Documents\LearnAgents\data_loader")
# print('File:',file)
# file_path = os.path.join(file, "data.txt")
loader = TextLoader(r"C:\Users\ad\Documents\LearnAgents\data_loader\data.txt")
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
# results = vector_db.similarity_search(
#     "The Legacy of a Shepherd",
#     k=2,
# )

template = """
     Context: {context}
     Answer the following question only after reading the context and if the answer is not 
     present in the context, Say I do no know:
    Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

llm = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),
)
# More Compact RunnableParallel

# Previous
# retriever_chain = (
#     RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
#     |prompt
#     |llm
#     |StrOutputParser()

# )

# Current
retriever_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()

)

print(retriever_chain.invoke("Who is Kaelin?"))