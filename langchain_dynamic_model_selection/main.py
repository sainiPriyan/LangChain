from dotenv import load_dotenv

# google_genai:gemini-2.5-flash-lite

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

basic_model = init_chat_model(model='google_genai:gemini-2.5-flash-lite')
advanced_model = init_chat_model(model='google_genai:gemini-3.1-pro-preview')

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler)->ModelRequest:
    message_count  = len(request.state['messages'])

    if message_count<3:
        model = basic_model

    else :
        model = advanced_model

    request.model = model

    return handler(request)    
    

agent = create_agent(model=basic_model,middleware=[dynamic_model_selection])

response = agent.invoke({
    'messages':[
        SystemMessage('You are a useful assistant.'),
        HumanMessage('How many are one plus one?'),
        HumanMessage('How many are one plus one?'),
        HumanMessage('How many are one plus one?')
    ]
})

print(response['messages'[-1].content])
print(response['messages'][-1].response_metadata['model_name'])
