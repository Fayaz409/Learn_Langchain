from langchain_community.document_loaders import TextLoader
import os
# Use this path format for reliability
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "data.txt")

loader = TextLoader(file_path)
text = loader.load()

print(text)

# WkipediaLoader

# from langchain_community.document_loaders import WikipediaLoader
# query  = ''
# loader = WikipediaLoader(query=f'{query}',load_max_docs=1)

# data = loader.load()
# print(data)


# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ('human', 'answer this question {question} from the context {context} if there is no context about the question respond with "I do not know"'),
#     ]
# )

# messages = prompt.format_messages(
#     name ='Babar Azam',
#     question = 'Who is Babar Azam?',
#     context = data,
# )

# CSV Loader

from langchain_community.document_loaders import CSVLoader
import os


csv_path = os.path.join(current_dir, "file.csv")

loader = CSVLoader(csv_path)

data = loader.load()
print('CSV Data',data)
