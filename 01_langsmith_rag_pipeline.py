import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

# 1. Import config để thiết lập các biến môi trường và cấu hình
import config
from qa_pairs import qa_pairs

# 2. Khởi tạo LLM và Embeddings từ OpenAI
llm = ChatOpenAI(model=config.DEFAULT_LLM_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=config.DEFAULT_EMBEDDING_MODEL)

# 3. Đọc dữ liệu (knowledge base) và chia nhỏ (chunking)
print("Loading and splitting knowledge base...")
loader = TextLoader("data/knowledge_base.txt", encoding="utf-8")
docs = loader.load()
# Chia văn bản với kích thước 500 ký tự và phần giao nhau là 50 ký tự
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 4. Tạo vector database (FAISS) từ các document chunks
print("Creating FAISS vector store...")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
# Cấu hình thành retriever (bộ tìm kiếm)
retriever = vectorstore.as_retriever()

# 5. Xây dựng prompt cho pipeline RAG
prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Xây dựng chuỗi (chain) thực thi của RAG: 
# Retriever lấy context -> ghép context và question vào prompt -> gửi cho LLM -> Phân tích kết quả đầu ra
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Sử dụng decorator @traceable để LangSmith tự động ghi lại quá trình xử lý hàm này
@traceable(name="rag-query")
def ask(chain, question: str) -> str:
    return chain.invoke(question)

if __name__ == "__main__":
    print(f"Running {len(qa_pairs)} queries through RAG pipeline...")
    # Chạy RAG qua tất cả 50 câu hỏi từ qa_pairs
    for idx, pair in enumerate(qa_pairs):
        q = pair["question"]
        ans = ask(rag_chain, q)
        print(f"Q{idx+1}: {q}")
        print(f"A: {ans[:50]}...\n")
    print("✅ Task 1 completed. Please check your LangSmith UI for the traces.")
