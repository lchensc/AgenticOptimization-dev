"""
Analysis Module - Deterministic metrics and AI-powered reasoning.

This module provides two layers of analysis:

1. **Deterministic Layer** (instant, free, reproducible):
   - compute_metrics() - All numerical metrics
   - Convergence analysis, pattern detection, efficiency metrics
   - Used for real-time monitoring and input to AI

2. **AI Reasoning Layer** (opt-in, costs money, strategic):
   - ai_analyze() - LLM-powered diagnosis and recommendations
   - Strategic insights beyond what deterministic metrics show
   - Returns actionable recommendations agent can execute

Design Principles:
- Deterministic is foundation (AI builds on top)
- AI analysis is optional (cost control)
- Structured output (agent can execute recommendations)
- Pure functions (no side effects)
"""

from .metrics import compute_metrics
from .ai_analysis import ai_analyze, AnalysisFocus

__all__ = [
    "compute_metrics",
    "ai_analyze",
    "AnalysisFocus",
]
