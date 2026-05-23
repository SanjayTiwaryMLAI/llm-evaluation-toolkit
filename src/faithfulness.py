"""Faithfulness Scoring — check if answers are grounded in context"""
from langchain_openai import ChatOpenAI


class FaithfulnessScorer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def score(self, answer: str, context: str) -> float:
        try:
            val = self.llm.invoke(
                f"Context: {context}\n\nAnswer: {answer}\n\n"
                f"Rate how faithfully the answer is grounded in the context (0.0-1.0, number only):"
            ).content.strip()
            return float(val)
        except Exception:
            return 0.5

    def batch_score(self, items: list[dict]) -> list[float]:
        return [self.score(i["answer"], i["context"]) for i in items]


if __name__ == "__main__":
    s = FaithfulnessScorer()
    print(s.score(
        answer="The Eiffel Tower is in Berlin.",
        context="The Eiffel Tower is a famous landmark in Paris, France.",
    ))
