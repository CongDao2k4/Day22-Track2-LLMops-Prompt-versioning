import os
import sys
import argparse

def run_step(script_name):
    print(f"\n{'='*50}\n🚀 Running {script_name}...\n{'='*50}")
    # Chạy từng file bằng command line
    os.system(f"{sys.executable} {script_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a specific step (1-4)")
    args = parser.parse_args()

    steps = {
        1: "01_langsmith_rag_pipeline.py",
        2: "02_prompt_hub_ab_routing.py",
        3: "03_ragas_evaluation.py",
        4: "04_guardrails_validator.py"
    }

    if args.step:
        # Nếu truyền tham số --step <number> thì chỉ chạy step đó
        run_step(steps[args.step])
    else:
        # Nếu không có tham số nào, chạy toàn bộ tuần tự
        for step_num in sorted(steps.keys()):
            run_step(steps[step_num])
