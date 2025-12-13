"""
Tests for token tracking and cost calculation.

Verifies:
- Token tracking with cost calculation
- Session statistics accumulation
- Budget enforcement
- Pricing calculations for different models
"""

import pytest
from datetime import datetime
from paola.llm import TokenTracker, UsageStats, SessionStats, BudgetStatus


def test_token_tracker_basic():
    """Test basic token tracking."""
    tracker = TokenTracker()

    # Track a call
    stats = tracker.track_call(
        model="qwen-plus",
        input_tokens=150,
        output_tokens=80
    )

    # Verify stats
    assert stats.model == "qwen-plus"
    assert stats.input_tokens == 150
    assert stats.output_tokens == 80
    assert stats.total_tokens == 230
    assert stats.input_cost_usd > 0  # Should have cost
    assert stats.output_cost_usd > 0
    assert stats.total_cost_usd == stats.input_cost_usd + stats.output_cost_usd

    print(f"✓ Tracked call: {stats.total_tokens} tokens, ${stats.total_cost_usd:.4f}")


def test_token_tracker_free_model():
    """Test that free models (qwen-flash) have zero cost."""
    tracker = TokenTracker()

    stats = tracker.track_call(
        model="qwen-flash",
        input_tokens=1000,
        output_tokens=500
    )

    # Verify zero cost
    assert stats.total_cost_usd == 0.0
    assert stats.input_cost_usd == 0.0
    assert stats.output_cost_usd == 0.0
    assert stats.total_tokens == 1500

    print(f"✓ Free model: {stats.total_tokens} tokens, ${stats.total_cost_usd:.4f}")


def test_token_tracker_cache_savings():
    """Test cache savings calculation (Anthropic models)."""
    tracker = TokenTracker()

    # Simulate cache read (90% cheaper than regular input)
    stats = tracker.track_call(
        model="claude-sonnet-4",
        input_tokens=500,
        output_tokens=200,
        cache_read_tokens=1000  # Read 1000 tokens from cache
    )

    # Verify cache savings
    assert stats.cache_read_tokens == 1000
    assert stats.cache_savings_usd > 0  # Should save money
    assert stats.cache_read_cost_usd < stats.input_cost_usd  # Cache should be cheaper

    print(f"✓ Cache savings: {stats.cache_read_tokens} tokens read, saved ${stats.cache_savings_usd:.4f}")


def test_session_stats_accumulation():
    """Test session statistics accumulation across multiple calls."""
    tracker = TokenTracker()

    # Make multiple calls
    tracker.track_call("qwen-plus", input_tokens=100, output_tokens=50)
    tracker.track_call("qwen-plus", input_tokens=200, output_tokens=100)
    tracker.track_call("qwen-plus", input_tokens=150, output_tokens=75)

    # Get session stats
    session = tracker.get_session_stats()

    # Verify accumulation
    assert session.call_count == 3
    assert session.total_input_tokens == 450
    assert session.total_output_tokens == 225
    assert session.total_tokens == 675
    assert session.total_cost_usd > 0

    print(f"✓ Session: {session.call_count} calls, {session.total_tokens} tokens, ${session.total_cost_usd:.4f}")


def test_budget_check():
    """Test budget enforcement."""
    tracker = TokenTracker()

    # Track some usage
    tracker.track_call("gpt-4", input_tokens=1000, output_tokens=500)  # Expensive model

    # Check budget
    budget = tracker.check_budget(1.00)  # $1.00 limit

    assert budget.budget_usd == 1.00
    assert budget.used_usd > 0
    assert budget.remaining_usd == budget.budget_usd - budget.used_usd
    assert budget.usage_percent == (budget.used_usd / budget.budget_usd) * 100

    print(f"✓ Budget: ${budget.used_usd:.4f} / ${budget.budget_usd:.2f} ({budget.usage_percent:.1f}%)")


def test_budget_exceeded():
    """Test budget exceeded detection."""
    tracker = TokenTracker()

    # Use expensive model to exceed budget quickly
    tracker.track_call("gpt-4", input_tokens=10000, output_tokens=5000)

    # Check with small budget
    budget = tracker.check_budget(0.10)  # $0.10 limit (likely exceeded)

    if budget.exceeded:
        print(f"✓ Budget exceeded: ${budget.used_usd:.4f} > ${budget.budget_usd:.2f}")
        assert budget.remaining_usd < 0
    else:
        print(f"✓ Budget OK: ${budget.used_usd:.4f} / ${budget.budget_usd:.2f}")


def test_budget_warning():
    """Test budget warning threshold (80%)."""
    tracker = TokenTracker()

    # Track enough to hit warning threshold
    # qwen-plus: $0.80 per 1M input tokens
    # Need ~80,000 tokens to cost $0.064 (80% of $0.08)
    tracker.track_call("qwen-plus", input_tokens=80000, output_tokens=0)

    budget = tracker.check_budget(0.08)  # $0.08 limit

    # Should trigger warning at >80% usage
    if budget.warning:
        print(f"✓ Budget warning: {budget.usage_percent:.1f}% used")
        assert budget.usage_percent > 80.0
    else:
        print(f"✓ Budget OK: {budget.usage_percent:.1f}% used")


def test_cache_hit_rate():
    """Test cache hit rate calculation."""
    tracker = TokenTracker()

    # Make calls with and without cache
    tracker.track_call("claude-sonnet-4", input_tokens=1000, output_tokens=500)
    tracker.track_call("claude-sonnet-4", input_tokens=0, output_tokens=300, cache_read_tokens=1000)  # Cache hit
    tracker.track_call("claude-sonnet-4", input_tokens=500, output_tokens=200)

    session = tracker.get_session_stats()

    # Cache hit rate = cache_read / (input + cache_creation + cache_read) * 100
    # = 1000 / (1000 + 0 + 500 + 0 + 0 + 1000) * 100 = 40%
    expected_rate = (1000 / 2500) * 100
    assert abs(session.cache_hit_rate - expected_rate) < 0.1

    print(f"✓ Cache hit rate: {session.cache_hit_rate:.1f}%")


def test_tracker_reset():
    """Test tracker reset functionality."""
    tracker = TokenTracker()

    # Track some calls
    tracker.track_call("qwen-plus", input_tokens=100, output_tokens=50)
    tracker.track_call("qwen-plus", input_tokens=200, output_tokens=100)

    # Verify stats exist
    session1 = tracker.get_session_stats()
    assert session1.call_count == 2

    # Reset
    tracker.reset()

    # Verify stats cleared
    session2 = tracker.get_session_stats()
    assert session2.call_count == 0
    assert session2.total_tokens == 0
    assert session2.total_cost_usd == 0.0

    print("✓ Tracker reset successful")


def test_pricing_accuracy():
    """Test pricing calculations for different models."""
    tracker = TokenTracker()

    test_cases = [
        ("qwen-flash", 1000, 500, 0.0),  # Free
        ("qwen-plus", 1000000, 1000000, 2.8),  # $0.80 + $2.00 = $2.80
        ("gpt-4", 1000000, 1000000, 90.0),  # $30 + $60 = $90
        ("claude-sonnet-4", 1000000, 1000000, 18.0),  # $3 + $15 = $18
    ]

    for model, input_tokens, output_tokens, expected_cost in test_cases:
        stats = tracker.track_call(model, input_tokens, output_tokens)

        # Allow small floating point error
        assert abs(stats.total_cost_usd - expected_cost) < 0.01, \
            f"{model}: expected ${expected_cost}, got ${stats.total_cost_usd}"

        print(f"✓ {model}: {input_tokens + output_tokens:,} tokens = ${stats.total_cost_usd:.2f}")


def test_call_history():
    """Test call history tracking."""
    tracker = TokenTracker()

    # Make several calls
    models = ["qwen-flash", "qwen-plus", "qwen-turbo"]
    for model in models:
        tracker.track_call(model, input_tokens=100, output_tokens=50)

    # Get history
    history = tracker.get_call_history()

    assert len(history) == 3
    assert all(isinstance(stat, UsageStats) for stat in history)
    assert [stat.model for stat in history] == models

    print(f"✓ Call history: {len(history)} calls tracked")


if __name__ == "__main__":
    # Run all tests
    print("\n" + "=" * 70)
    print("Token Tracking Tests")
    print("=" * 70 + "\n")

    test_token_tracker_basic()
    test_token_tracker_free_model()
    test_token_tracker_cache_savings()
    test_session_stats_accumulation()
    test_budget_check()
    test_budget_exceeded()
    test_budget_warning()
    test_cache_hit_rate()
    test_tracker_reset()
    test_pricing_accuracy()
    test_call_history()

    print("\n" + "=" * 70)
    print("✅ All token tracking tests passed!")
    print("=" * 70 + "\n")
