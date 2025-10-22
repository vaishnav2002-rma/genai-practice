import ollama 

response = ollama.chat(model="llama3.1", messages=[
    {'role': 'user', 'content':'Explain RAG in simple terms'}
])

print(response['message']['content'])