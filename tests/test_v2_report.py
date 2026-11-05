"""Paired user analysis preserves the protocol's user weighting."""
import pytest

from scripts.report_v2_study import paired_user_difference


def row(user, request, scores):
    return {"task": "search", "user_raw": user, "request_id": request, "cohort": "same_service",
            "item_raw": [10, 20], "labels": [1, 0], "scores": scores}


def test_user_bootstrap_is_paired_and_weights_users_equally():
    # User a has two requests and user b one: averaging over requests would
    # over-weight a. Left-to-right change only occurs for b.
    left = [row("a", "1", [1, 0]), row("a", "2", [1, 0]), row("b", "3", [0, 1])]
    right = [row("a", "1", [1, 0]), row("a", "2", [1, 0]), row("b", "3", [1, 0])]
    result = paired_user_difference(left, right)
    assert result["users"] == 2
    assert result["mean_delta"] == pytest.approx((1-1/__import__("math").log2(3))/2)
    assert paired_user_difference(left, right) == result
    assert paired_user_difference(right, left)["mean_delta"] == pytest.approx(-result["mean_delta"])
    with pytest.raises(ValueError, match="identical search"):
        paired_user_difference(left, right[:1])
