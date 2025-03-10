from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "data.txt")



loader = TextLoader(file_path)
data = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator='.',
    length_function=len
)

texts = splitter.split_documents(data)
print(len(texts))
print('Charater Text Splitter',texts)


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter_ = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=['.','!','?'],
    length_function=len
)

texts_ = splitter_.split_documents(data)
print('Texts Recursive Character Text Splitter',texts_)
print(len(texts))