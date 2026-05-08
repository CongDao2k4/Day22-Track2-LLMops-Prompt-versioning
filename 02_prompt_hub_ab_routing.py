import os
import hashlib
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import config
from qa_pairs import qa_pairs

# 1. Khởi tạo LangSmith client và LLM
client = Client(api_key=config.LANGSMITH_API_KEY)
llm = ChatOpenAI(model=config.DEFAULT_LLM_MODEL, temperature=0)

# 2. Xây dựng 2 phiên bản Prompt khác nhau
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be very concise and direct in your answers."),
    ("user", "{question}")
])

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", "You are an expert AI tutor. Provide structured, detailed, and polite answers. Use bullet points if necessary."),
    ("user", "{question}")
])

prompt_name = "day22-rag-prompt-v"

# 3. Đẩy 2 prompts này lên LangSmith Prompt Hub
try:
    print("Pushing prompts to Hub...")
    client.push_prompt(f"{prompt_name}1", object=PROMPT_V1, description="Concise version")
    client.push_prompt(f"{prompt_name}2", object=PROMPT_V2, description="Detailed version")
    print("✅ Prompts pushed successfully.")
except Exception as e:
    print(f"Could not push prompts: {e}")

# 4. Hàm A/B Routing sử dụng MD5 hash để chia luồng 50/50 dựa trên request_id
def get_prompt_version(request_id: str) -> str:
    h = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    # Lấy bản chẵn lẻ (50/50)
    return f"{prompt_name}1" if h % 2 == 0 else f"{prompt_name}2"

if __name__ == "__main__":
    print("Starting A/B Routing test...")
    # Chạy 50 câu hỏi qua hệ thống routing này
    for idx, pair in enumerate(qa_pairs):
        q = pair["question"]
        request_id = f"req-{idx}"
        
        # Lựa chọn version dựa vào request_id
        version = get_prompt_version(request_id)
        
        # Tải Prompt đã lưu từ Hub (có xử lý fallback phòng trường hợp lỗi kết nối)
        try:
            pulled_prompt = client.pull_prompt(version)
        except Exception:
            pulled_prompt = PROMPT_V1 if "v1" in version else PROMPT_V2
            
        # Tạo chuỗi xử lý và chạy
        chain = pulled_prompt | llm | StrOutputParser()
        ans = chain.invoke({"question": q})
        
        print(f"[{version}] Q: {q} -> A: {ans[:40]}...")
