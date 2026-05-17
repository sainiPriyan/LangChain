from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model('google_genai:gemini-2.5-flash-lite')

message = {
    'role': 'user',
    'content':[
        {'type':'text', 'text':'Describe the content of this image'},
        {'type':'image', 'url':'https://platform.vox.com/wp-content/uploads/sites/2/chorus/uploads/chorus_asset/file/15231691/453801468.0.0.1421786380.jpg'}

    ]
}

response = model.invoke([message])

print(response.content)