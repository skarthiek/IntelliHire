import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np

GEMINI_API_KEY = "AIzaSyAwFoFbeJ7Nh5mlSCz_TeWAUforAA_h2Dc"
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_embedding(text):
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

def create_vectorstore(text, chunk_size=500, chunk_overlap=100):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text])
    texts = [doc.page_content for doc in docs]
    embeddings = [get_gemini_embedding(text) for text in texts]
    import faiss
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    embeddings_array = np.array(embeddings).astype('float32')
    index.add(embeddings_array)
    class SimpleVectorStore:
        def __init__(self, index, docs):
            self.index = index
            self.docs = docs
        def similarity_search(self, query, k=30):
            query_embedding = get_gemini_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            distances, indices = self.index.search(query_vector, k)
            return [self.docs[i] for i in indices[0] if i < len(self.docs)]
    return SimpleVectorStore(index, docs)

def get_answer(db, query, k=30):
    relevant_docs = db.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    prompt = f"""Answer based on the context below. Provide the answer in a structured format such as a list or table if applicable.\n\n{context}\n\nQuestion: {query}\nAnswer:"""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()
