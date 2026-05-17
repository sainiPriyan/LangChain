import requests
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model='google_genai:gemini-2.5-flash-lite',
    temperature = 0.2
)


conversation = [SystemMessage('You are a helpful AI Assistant for questions regarding programming.'),
                HumanMessage('What is Python?'),
                AIMessage('Python is an interpreted programming language.'),
                HumanMessage('When was it released?')]

# response = model.invoke('Hello, What is Python?')

# response = model.invoke(conversation)

# print(response.content)

for chunk in model.stream('Hello, What is Python?'):
    print(chunk.text,end='', flush=True)





