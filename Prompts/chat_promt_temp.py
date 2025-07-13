from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

chat_promt = ChatPromptTemplate([
    ('system' , "You are a {domain} expert."),
    ('human' , "Tell me about {topic} in {language} language."),
])

prompt = chat_promt.invoke({"domain" : "Cricket" , "topic" : "Draw" , "language" : "English"})

print(prompt)