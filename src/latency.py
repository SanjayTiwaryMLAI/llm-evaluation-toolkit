"""Latency Benchmarking across LLM providers"""
import time
import statistics
from openai import OpenAI


class LatencyBenchmark:
    def __init__(self, runs: int = 5):
        self.client  = OpenAI()
        self.runs    = runs
        self.results = {}

    def _measure(self, model: str, prompt: str) -> dict:
        latencies = []
        for _ in range(self.runs):
            start = time.time()
            self.client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            latencies.append(time.time() - start)
        return {
            "p50":  statistics.median(latencies),
            "p95":  sorted(latencies)[int(0.95 * len(latencies))],
            "mean": statistics.mean(latencies),
        }

    def run(self, models: list[str], prompt: str):
        for model in models:
            print(f"  Benchmarking {model} ...")
            self.results[model] = self._measure(model, prompt)

    def report(self):
        print(f"\n{'Model':<20} {'P50(s)':<10} {'P95(s)':<10} {'Mean(s)':<10}")
        print("-" * 50)
        for m, r in self.results.items():
            print(f"{m:<20} {r['p50']:<10.2f} {r['p95']:<10.2f} {r['mean']:<10.2f}")


if __name__ == "__main__":
    b = LatencyBenchmark(runs=3)
    b.run(["gpt-4o-mini", "gpt-4o"], "Explain RAG in 3 sentences.")
    b.report()
