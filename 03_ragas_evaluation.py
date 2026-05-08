import os
import json
import warnings
from datasets import Dataset

warnings.filterwarnings("ignore")

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config
from qa_pairs import qa_pairs

# 1. Khởi tạo mô hình đánh giá
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model=config.DEFAULT_EMBEDDING_MODEL)

# 2. Định nghĩa hàm load dataset để test
def load_ragas_dataset():
    # RAGAS yêu cầu dataset có cấu trúc: question, answer, contexts, ground_truth
    # Đây là dữ liệu mẫu giả lập dựa vào qa_pairs. Trong thực tế, bạn sẽ lấy "answer" và "contexts" từ kết quả chạy Pipeline
    data = {
        "question": [p["question"] for p in qa_pairs[:5]],
        "answer": ["Machine learning is a subset of AI.", "Overfitting means the model memorizes the data.", "Bias is error from wrong assumptions.", "Regularization adds a penalty to error.", "Cross-validation is a resampling method."],
        "contexts": [
            ["Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data."],
            ["Overfitting occurs when a machine learning model learns the training data too well, including noise and details."],
            ["Bias is the simplifying assumptions made by a model to make the target function easier to learn."],
            ["Regularization involves adding a penalty term to the loss function to prevent overfitting."],
            ["Cross validation is a technique used to evaluate machine learning models on a limited data sample."]
        ],
        "ground_truth": [p["reference"] for p in qa_pairs[:5]]
    }
    return Dataset.from_dict(data)

if __name__ == "__main__":
    print("Preparing RAGAS Evaluation Dataset...")
    dataset = load_ragas_dataset()
    
    print("Running Evaluation (this might take a few minutes)...")
    try:
        # 3. Chạy RAGAS evaluate với các metric được yêu cầu
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=llm,
            embeddings=embeddings
        )
        
        print("\n📊 RAGAS Evaluation Results:")
        print(result)
        
        # 4. Lưu kết quả ra file
        os.makedirs("data", exist_ok=True)
        with open("data/ragas_report.json", "w") as f:
            json.dump(result, f, indent=4)
        print("✅ Results saved to data/ragas_report.json")
        
        if result["faithfulness"] >= 0.8:
            print("✅ Metric 'faithfulness' meets the target >= 0.8")
        else:
            print("⚠️ Metric 'faithfulness' is below target 0.8")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
