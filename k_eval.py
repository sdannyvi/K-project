"""
Experimental conditions for evaluating responses against a K-aligned rubric (future work).

Defines symbolic labels for ablations: no retrieval, grounded RAG, and a negative
control with mismatched chunks. Extend when you add human or model-based scoring.

Update when the formal rubric dimensions (e.g. inquiry vs. prescription) are fixed.
"""

# Ordered tuple for reporting or automated sweeps.
EVAL_CONDITIONS: tuple[str, ...] = (
    "baseline_no_rag",
    "rag_k_texts",
    "rag_wrong_chunks_control",
)


def describe_wrong_chunk_control() -> str:
    """
    Return a short description of the shuffled/wrong-chunk negative control.

    Returns:
        Human-readable one-line explanation for experiment docs.
    """
    return (
        "Retrieve chunks with intentionally poor relevance (e.g. shuffled "
        "indices) to test confabulation versus grounding."
    )
