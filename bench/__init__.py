"""Benchmark harness for spec §6.8 — the honest test.

**This package is not part of the substrate.** It exists to answer one
question: does CANDOR actually beat the boring alternative? To ask that fairly
the baseline needs text embeddings and an LLM, both of which the substrate
forbids (§0 non-goals, "no text embeddings anywhere in the core").

The separation is physical and one-way:

    bench/  ->  candor/        allowed (the harness drives the system)
    candor/ ->  bench/         never; asserted by tests/unit/test_audit.py

Nothing in `src/candor/` imports this package, and nothing here is imported at
substrate runtime. Embeddings live on this side of the line, permanently.
"""

__all__ = ["corpus", "ollama", "metrics"]
