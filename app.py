import dotenv
import os
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import json
print('Hello, World!')
from langchain_community.document_loaders import YoutubeLoader
# print('SECRET_KEY =', os.getenv('SECRET_KEY'))

# loader = YoutubeLoader.from_youtube_url(
#     "https://www.youtube.com/watch?v=QsYGlZkevEg", add_video_info=False
# )

# print(loader.load())

conn = sqlite3.connect('example.db')

json_data = {
    "users": [
      {
        "id": 1,
        "name": "John Doe",
        "email": "john@example.com",
        "age": 30,
      },
      {
        "id": 2,
        "name": "Jane Smith",
        "email": "jane@example.com",
        "age": 28,
      }
    ],
    "orders": [
      {
        "order_id": 1001,
        "user_id": 1,
        "items": [
          {"product": "Laptop", "price": 999.99},
          {"product": "Mouse", "price": 29.99}
        ],
        "total": 1029.98
      }
    ]
  }

# conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT, age INTEGER)")
# for item in json_data['users']:
#     conn.execute("INSERT INTO users (id, name, email, age) VALUES (?, ?, ?, ?)",
#                  (item["id"], item["name"], item["email"], item["age"]))
conn.commit()
curser = conn.cursor()
curser.execute("SELECT * FROM users")
print(curser.fetchall())
conn.close()

