from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import SystemMessage , HumanMessage, AIMessage
from dotenv import load_dotenv  
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

chat_histroy = [
    SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_histroy.append(HumanMessage(content = user_input))
    if(user_input == "exit"):
        break
    res = model.invoke(chat_histroy)
    print("AI: " , res.content)
    chat_histroy.append(AIMessage(content = res.content))
    
print(chat_histroy)    
    