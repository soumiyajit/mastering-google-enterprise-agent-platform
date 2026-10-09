import os
import time
import pandas as pd
from typing import Dict, Any, List
from google import genai
from google.genai import types

# 1. Initialize Client
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "gemini-galacticos")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# 2. Define the Tool Schema globally before functions
db_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="query_enterprise_database",
            description="Queries enterprise SQL database and returns matching records.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query_string": types.Schema(type="STRING", description="Search query string"),
                    "limit": types.Schema(type="INTEGER", description="Max returned rows"),
                    "filter_status": types.Schema(type="STRING", description="Status filter")
                },
                required=["query_string"]
            )
        )
    ]
)

# 3. Define Models and Prompts
MODELS_TO_BENCHMARK = ["gemini-2.5-pro", "gemini-2.5-flash"]
TEST_PROMPTS = [
    {
        "prompt": "Find all records related to quarterly compliance reports, up to 25 items, with PENDING status.",
        "expected_args": {"query_string": "quarterly compliance reports", "limit": 25, "filter_status": "PENDING"}
    }
]
NUM_ITERATIONS = 3

# 4. Define Benchmark Iteration Function
def run_benchmark_iteration(model_id: str, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
    config = types.GenerateContentConfig(
        temperature=0.0,
        tools=[db_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    )

    start_time = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt_data["prompt"],
            config=config
        )
        end_time = time.perf_counter()
        latency_seconds = end_time - start_time

        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count if usage else 0
        output_tokens = usage.candidates_token_count if usage else 0
        tokens_per_sec = output_tokens / latency_seconds if latency_seconds > 0 else 0

        # Validate Tool Call Accuracy
        function_calls = response.function_calls
        accuracy_score = 0.0
        if function_calls:
            first_call = function_calls[0]
            extracted_args = first_call.args
            accuracy_score = 1.0
            for key, expected_val in prompt_data["expected_args"].items():
                if extracted_args.get(key) != expected_val:
                    accuracy_score = 0.0
                    break

        return {
            "Model": model_id,
            "Latency (s)": round(latency_seconds, 3),
            "Input Tokens": input_tokens,
            "Output Tokens": output_tokens,
            "Tokens/Sec": round(tokens_per_sec, 2),
            "Invocation Accuracy": accuracy_score
        }

    except Exception as e:
        print(f"Error benchmarking {model_id}: {e}")
        return {
            "Model": model_id,
            "Latency (s)": 0.0,
            "Input Tokens": 0,
            "Output Tokens": 0,
            "Tokens/Sec": 0.0,
            "Invocation Accuracy": 0.0
        }

# 5. Main Execution Loop
def execute_benchmarks():
    print("--- Starting Model Garden Tool Invocation Benchmark ---")
    results: List[Dict[str, Any]] = []

    for model in MODELS_TO_BENCHMARK:
        print(f"Benchmarking Model: {model}...")
        for test_case in TEST_PROMPTS:
            for _ in range(NUM_ITERATIONS):
                res = run_benchmark_iteration(model, test_case)
                results.append(res)
                time.sleep(0.5)

    df = pd.DataFrame(results)
    summary = df.groupby("Model").agg({
        "Latency (s)": "mean",
        "Tokens/Sec": "mean",
        "Invocation Accuracy": "mean"
    }).reset_index()

    summary["Invocation Accuracy (%)"] = summary["Invocation Accuracy"] * 100
    summary.drop(columns=["Invocation Accuracy"], inplace=True)

    print("\n================ Aggregated Benchmark Results ================")
    print(summary.to_string(index=False))
    print("==============================================================")

if __name__ == "__main__":
    execute_benchmarks()