from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    'I love The Beatles.',
    'Beetles represent the largest insect group.',
    'An apple a day keeps the doctor away!',
    'My doctor uses an Apple phone.',
    'Orange is my favorite fruit.',
    'That is an orange IPhone'
]

vector_store = FAISS.from_texts(
    texts,
    embedding=embeddings
)

print(vector_store.similarity_search('I have an orange colored electric guitar',k=len(texts)))