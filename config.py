import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# Lấy các biến môi trường
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGCHAIN_PROJECT", "day22-langsmith-lab")

# Cấu hình LangSmith tracing (bắt buộc phải bật để trace)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Các model mặc định
DEFAULT_LLM_MODEL = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

if __name__ == "__main__":
    print("✅ Config loaded successfully")
    print(f"   LangSmith project : {LANGSMITH_PROJECT}")
    print(f"   Default LLM model : {DEFAULT_LLM_MODEL}")
    print(f"   Embedding model   : {DEFAULT_EMBEDDING_MODEL}")
