from langchain_core.messages import SystemMessage , AIMessage , HumanMessage
from langchain_core.prompts import ChatPromptTemplate 
from dotenv import load_dotenv
load_dotenv()

# chat template
chat_template = ChatPromptTemplate([
    ('system' , 'You are helpful customer support agent'),
    ('placeholder' , '{chat_history}'),
    ('human' , '{query}')
])

# Load chat histroy
chat_histroy = []

with open('./Prompts/chat_history.txt') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if line == 'human' and i + 1 < len(lines):
            content = lines[i + 1].strip()
            chat_histroy.append(HumanMessage(content=content))
        elif line == 'ai' and i + 1 < len(lines):
            content = lines[i + 1].strip()
            chat_histroy.append(AIMessage(content=content))

print(chat_histroy)
# Prompt

promt = chat_template.invoke({'chat_history' : chat_histroy , 'query' : "Where is my refund ?"})

print(promt)