"""Hallucination Detection — NLI-based + GPT-judge"""
from transformers import pipeline
from langchain_openai import ChatOpenAI


class HallucinationDetector:
    def __init__(self, method: str = "nli"):
        self.method = method
        if method == "nli":
            self.nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
        else:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def score_nli(self, claim: str, context: str) -> float:
        r = self.nli(f"{context} [SEP] {claim}")[0]
        if r["label"] == "ENTAILMENT":   return 1.0 - r["score"]
        if r["label"] == "CONTRADICTION": return r["score"]
        return 0.5

    def score_gpt(self, claim: str, context: str) -> float:
        try:
            val = self.llm.invoke(
                f"Context: {context}\nClaim: {claim}\n"
                f"Rate how well the context supports the claim (0.0-1.0, number only):"
            ).content.strip()
            return 1.0 - float(val)
        except Exception:
            return 0.5

    def score(self, claim: str, context: str) -> float:
        return self.score_nli(claim, context) if self.method == "nli" else self.score_gpt(claim, context)

    def batch_score(self, pairs: list[dict]) -> list[float]:
        return [self.score(p["claim"], p["context"]) for p in pairs]


if __name__ == "__main__":
    d = HallucinationDetector(method="gpt")
    print(d.score("The Eiffel Tower is in Berlin.", "The Eiffel Tower is in Paris, France."))
