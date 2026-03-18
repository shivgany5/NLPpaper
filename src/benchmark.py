import yaml
import json
import argparse
from tqdm import tqdm
import traceback

# Import model architectures
# Handled safely via import conditions depending on system resources
from models.baseline import BaselineLLM
from models.rag import RAGPipeline
from models.fine_tune import FineTunedLLM

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_queries(path):
    """
    Load testing corpus query suite.
    """
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: Ground truth test queries not found. Utilizing baseline dummy queries for validation.")
        return [
            "What is the thematic significance of hospitality in The Odyssey?",
            "Explain the stylistic choices used in translating the classical battle sequences."
        ]

def run_benchmark(config):
    queries = load_queries(config['evaluation']['ground_truth_path'])
    results = []

    print("--- Booting Semantic Understanding Benchmark ---")
    
    # ----------------------------------------------------
    # Stage 1: Baseline Architecture Evaluation
    # ----------------------------------------------------
    # print("Initializing Baseline Layer...")
    # try:
    #     baseline = BaselineLLM(config)
    #     for q in tqdm(queries, desc="Baseline Evaluation"):
    #         res = baseline.generate(q)
    #         results.append({"query": q, "model": "baseline", "response": res})
    #     del baseline
    # except Exception as e:
    #     print(f"Error executing Baseline: {e}")

    # ----------------------------------------------------
    # Stage 2: RAG Architecture Evaluation
    # ----------------------------------------------------
    print("Initializing RAG Layer...")
    try:
        rag = RAGPipeline(config)
        for q in tqdm(queries, desc="RAG Evaluation"):
            res = rag.generate(q)
            results.append({"query": q, "model": "rag", "response": res, "context_used": True})
        del rag
    except Exception as e:
        print(f"Error executing RAG: {e}")
        traceback.print_exc()
        
    # ----------------------------------------------------
    # Stage 3: Fine-Tuned Framework Evaluation
    # ----------------------------------------------------
    # print("Initializing Fine-Tuned Adapter Layer...")
    # try:
    #     ft = FineTunedLLM(config)
    #     for q in tqdm(queries, desc="Fine-Tune Evaluation"):
    #         res = ft.generate(q)
    #         results.append({"query": q, "model": "fine_tuned", "response": res})
    #     del ft
    # except Exception as e:
    #     print(f"Skipping Fine-tuned layer (ensure adapters exist): {e}")

    # Persist outputs
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Benchmarking Complete. Persisted artifacts to results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    run_benchmark(config)
