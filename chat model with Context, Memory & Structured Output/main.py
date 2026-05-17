import requests
from dotenv import load_dotenv
from dataclasses import dataclass
import os

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat():
    fun_summary: str 
    humidity: int 
    temperature_celcius: float 
    temperature_fahrenheit: float 

@tool('locate_user', description='look up a user\'s city based on context')   
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id :
        case 'ABC123':
            return 'Jaipur'
        case 'XYZ123':
            return 'New Delhi'
        case _ :
            return 'Unknown'


@tool('get_weather', description='return weather information for a given city', return_direct=False)
def get_weather(city: str) -> str:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

model = init_chat_model('google_genai:gemini-2.5-flash-lite',temperature = 0.3)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather,locate_user],
    system_prompt='You are a helpful weather assistant agent, who always cracks jokes and is humorous while also remaining helpful',
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {'configurable':{'thread_id':'1'}}

response = agent.invoke({
    'messages': [{'role':'user', 'content':'What is the weather like?'}]},
     config=config,
     context=Context(user_id='XYZ123')
     )

# print(response)
# print('---------------')
# print(response['messages'][-1].content)

print(response['structured_response'])
print(response['structured_response'].fun_summary)
print(response['structured_response'].temperature_celcius)

config = {'configurable':{'thread_id':'2'}}

response = agent.invoke({
    'messages': [{'role':'user', 'content':'and is this usual?'}]},
     config=config,
     context=Context(user_id='ABC123')
     )

print(response['structured_response'])
print(response['structured_response'].fun_summary)
print(response['structured_response'].temperature_celcius)