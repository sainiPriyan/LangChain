from dataclasses import dataclass
from dotenv import load_dotenv

# google_genai:gemini-2.5-flash-lite

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from time import time

load_dotenv()

class HooksDemo(AgentMiddleware):

    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state, runtime):
        self.start_time = time()
        print('Agent triggered')

    def before_model(self, state, runtime):
        print('before_model')

    def after_model(self, state, runtime):
        self.start_time = time()
        print('after_model')   

    def after_agent(self, state, runtime):
        print('after_agent:',time()-self.start_time)     

agent = create_agent('google_genai:gemini-2.5-flash-lite', middleware= [HooksDemo()])           

response = agent.invoke({
    'messages':[
        SystemMessage('You are a helpful assistant'),
        HumanMessage('What is PCA?')
    ]
})

print(response['messages'][-1].content)