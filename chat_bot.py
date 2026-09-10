import ollama

response = ollama.chat(
    model='LiquidAI/lfm2.5-1.2b-instruct',
    messages=[
        {'role':'user','content':'Write a langchain code which implements a end to end rag pipline of fetching user data from the particular external sourse of database.'}
    ]
)
print(response['message']['content'])