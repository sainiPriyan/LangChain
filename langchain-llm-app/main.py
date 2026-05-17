import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool('get_weather', description='return weather information for a given city', return_direct=False)
def get_weather(city: str) -> str:
    response = requests.get(f'https://wttr.in/{city}?format=j1')
    return response.json()

agent = create_agent(
    model='google_genai:gemini-2.5-flash-lite',
    tools=[get_weather],
    system_prompt='You are a helpful weather assistant agent, who always cracks jokes and is humorous while also remaining helpful'
)

response = agent.invoke({
    'messages': [{'role':'user', 'content':'What is the weather like in Jaipur?'}]
})

print(response)
print('---------------')
print(response['messages'][-1].content)
