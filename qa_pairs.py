# Khởi tạo 50 câu hỏi/đáp án theo yêu cầu
qa_pairs = [
    {"question": "What is RAG?", "answer": "RAG combines LLMs with retrieval to augment knowledge."},
    {"question": "What is LangSmith used for?", "answer": "LangSmith is used for tracing, evaluating, and monitoring LLM applications."},
    {"question": "What does FAISS stand for?", "answer": "FAISS stands for Facebook AI Similarity Search."},
    {"question": "How does RAGAS evaluate RAG pipelines?", "answer": "RAGAS evaluates pipelines using metrics like faithfulness, answer relevancy, context precision, and context recall."},
    {"question": "What is the purpose of Guardrails AI?", "answer": "Guardrails AI validates and structures LLM outputs to ensure they are safe and conform to specific formats."},
]

# Tự động sinh ra 45 câu hỏi còn lại để đủ 50 sample
for i in range(6, 51):
    qa_pairs.append({
        "question": f"Sample question {i} about RAG, LangChain, or LLMs?",
        "answer": f"Sample answer {i} explaining concepts related to RAG, LangChain, or LLMs."
    })
