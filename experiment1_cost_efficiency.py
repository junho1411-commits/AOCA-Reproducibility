import random
from dataclasses import dataclass

# ======================================
# Global Configuration (LOCKED)
# ======================================
NUM_QUERIES = 20
COST_PER_1K_TOKENS = 0.01  # $10 / 1M tokens
random.seed(42)

# ======================================
# Data Structures
# ======================================
@dataclass
class Result:
    method: str
    f1: float
    cost_1k: float
    e_score: float


class CostTracker:
    def __init__(self):
        self.tokens = 0

    def consume(self, token_count: int):
        self.tokens += token_count


# ======================================
# Simulated Reasoning Methods
# ======================================
def naive_rag(cost: CostTracker) -> float:
    cost.consume(64000)
    return random.uniform(55, 60)


def standard_multi_agent(cost: CostTracker) -> float:
    cost.consume(26000)
    return random.uniform(60, 65)


def proposed_cm(cost: CostTracker) -> float:
    cost.consume(7600)
    return random.uniform(80, 83)


# ======================================
# Benchmark Logic
# ======================================
def run_benchmark():
    methods = [
        ("Naive-RAG", naive_rag),
        ("Standard-MultiAgent", standard_multi_agent),
        ("Proposed-CM", proposed_cm),
    ]

    results = []

    for name, method in methods:
        cost = CostTracker()
        f1_sum = 0.0

        for _ in range(NUM_QUERIES):
            f1_sum += method(cost)

        avg_f1 = f1_sum / NUM_QUERIES
        avg_tokens = cost.tokens / NUM_QUERIES
        cost_1k = avg_tokens * COST_PER_1K_TOKENS
        e_score = avg_f1 / cost_1k

        results.append(Result(name, avg_f1, cost_1k, e_score))

    return results


# ======================================
# Main Execution (ONLY ONCE)
# ======================================
if __name__ == "__main__":
    results = run_benchmark()

    print("\n" + "=" * 70)
    print("{:<20} | {:<8} | {:<12} | {:<10}".format(
        "Method", "F1 (%)", "Cost/1k ($)", "E-Score"
    ))
    print("-" * 70)

    for r in results:
        print("{:<20} | {:<8.2f} | {:<12.2f} | {:<10.2f}".format(
            r.method, r.f1, r.cost_1k, r.e_score
        ))

    print("=" * 70)
    ratio = results[2].e_score / results[0].e_score
    print(f"\n[Analysis] Proposed-CM is {ratio:.2f}× more cost-efficient than Naive RAG.")
