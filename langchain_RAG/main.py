from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    'I love Apples',
    'Orange is my favorite fruit',
    'I like guavas',
    'I hate bananas',
    'I really don\'t like Grapes',
    'I think water-melons sucks'
]

vector_store = FAISS.from_texts(
    texts,
    embedding=embeddings
)

print(vector_store.similarity_search('What fruits does the person like?',k=3))
print('---------------')
print(vector_store.similarity_search('What fruits does the person dislike?',k=3))
print('---------------')

retriever = vector_store.as_retriever(search_kwargs={'k':3})

retriever_tool = create_retriever_tool(retriever, name='kn_search', 
                                       description='Search the small product/fruit knowledge base from information')

agent = create_agent(
    model='google_genai:gemini-2.5-flash-lite',
    tools=[retriever_tool],
    system_prompt=('You are a helpful assistant for a fruit knowledge base, '
                   'first you call the kb_search tool to retrieve context, then answer succinctly.'
                   'Maybe you have to use it multiple times before answering it'
                   ),

)

result = agent.invoke({
    'messages': [{'role':'user',
                  'content':'what three fruits does the person like and what three fruits does the person dislike'}]
})

print(result)
print(result['messages'][-1].content)