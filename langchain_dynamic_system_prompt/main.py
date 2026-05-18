from dotenv import load_dotenv

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

load_dotenv()

@dataclass
class Context:
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = 'You are a helpful and very concise assistant.'

    match user_role:
        case 'expert':
            return f'{base_prompt} Provide detail and technical responses'
        case 'beginner':
            return f'{base_prompt} Keep your explanations simple and basic'
        case 'child':
            return f'{base_prompt} Explain everything as if you were explaining to a five year old kid, literally'
        case _:
            return base_prompt
        
agent = create_agent(model='google_genai:gemini-2.5-flash-lite',
                     middleware=[user_role_prompt],
                     context_schema=Context)

response = agent.invoke({'messages':[{'role':'user','content':'Explain nuclear fission'}]},
                         context=Context(user_role='expert'))      

print(response)  
print('==================================')
print(response['messages'][-1].content)
