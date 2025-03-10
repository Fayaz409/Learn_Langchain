import sqlite3
import json

# Connect to database
conn = sqlite3.connect('example.db')
c = conn.cursor()

# Create tables
c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        age INTEGER,
        street TEXT,
        city TEXT,
        state TEXT,
        zip TEXT
    )
''')

c.execute('''
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY,
        user_id INTEGER,
        items TEXT,
        total REAL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
''')

# Load JSON data
try:
    with open(r'C:\Users\ad\Documents\LearnAgents\example.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("JSON file not found!")
except json.JSONDecodeError:
    print("Invalid JSON format!")

# # Insert users
# for user in data.get('users', []):
#     c.execute('''
#         INSERT INTO users (id, name, email, age, street, city, state, zip)
#         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#     ''', (
#         user['id'],
#         user['name'],
#         user['email'],
#         user['age'],
#         user['address']['street'],
#         user['address']['city'],
#         user['address']['state'],
#         user['address']['zip']
#     ))

# # Insert orders
# for order in data.get('orders', []):
#     c.execute('''
#         INSERT INTO orders (order_id, user_id, items, total)
#         VALUES (?, ?, ?, ?)
#     ''', (
#         order['order_id'],
#         order['user_id'],
#         json.dumps(order['items']),
#         order['total']
#     ))

conn.commit()

c.execute('select * from users')
print(c.fetchall())
c.execute('select * from orders')
print(c.fetchall())
c.close()