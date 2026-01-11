# =========================
# Configuration
# =========================

CONTEXT_SIZES = [2000, 8000, 20000, 50000, 100000, 200000]

# Token reduction ratios per architecture
ARCHITECTURES = {
    "Naive-RAG": 1.0,
    "Standard-MultiAgent": 0.45,
    "Proposed-CM": 0.15,
}

# Normalized simulation constants
LATENCY_PER_1K_TOKENS_MS = 150.0     # baseline latency
COST_PER_1K_TOKENS_USD = 0.01        # normalized token cost


# =========================
# Core computation
# =========================

def compute_latency(tokens: int) -> float:
    """Latency scales linearly with token count."""
    return (tokens / 1000.0) * LATENCY_PER_1K_TOKENS_MS


def compute_cost(tokens: int) -> float:
    """Cost per query based on normalized token pricing."""
    return (tokens / 1000.0) * COST_PER_1K_TOKENS_USD


# =========================
# Main experiment
# =========================

def main():
    print("\n" + "=" * 110)
    print(f"{'Context Size':<13} {'Method':<22} {'Avg Tokens':<13} {'Latency (ms)':<15} {'Cost / Query ($)':<18}")
    print("-" * 110)

    for ctx in CONTEXT_SIZES:
        for method, ratio in ARCHITECTURES.items():
            avg_tokens = int(ctx * ratio)
            latency = compute_latency(avg_tokens)
            cost = compute_cost(avg_tokens)

            print(
                f"{ctx:<13} "
                f"{method:<22} "
                f"{avg_tokens:<13} "
                f"{latency:<15.2f} "
                f"{cost:<18.4f}"
            )

    print("=" * 110)
    print("\nNote: Latency and cost are estimated using a normalized simulation model.")
    print("Results reflect relative scalability trends rather than absolute runtime measurements.\n")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    main()
