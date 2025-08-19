import pytest
from main import ab_test, sample_size, bayesian_ab_test

def test_sample_size_basic():
    n=sample_size(p=0.1, mde=0.02, alpha=0.05, power=0.8)
    assert n > 0
    assert isinstance(n, int)

def test_sample_size_invalid_baseline():
    with pytest.raises(ValueError):
        sample_size(p=-0.1, mde=0.02)

def test_sample_size_invalid_mde():
    with pytest.raises(ValueError):
        sample_size(p=0.1, mde=-0.01)

def test_ab_test_basic():
    result=ab_test(visitors_a=1000, conversions_a=120, visitors_b=1000, conversions_b=150)
    assert "cr_a" in result
    assert "cr_b" in result
    assert "p_value" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert isinstance(result["p_value"], float)

def test_ab_test_invalid_visitors():
    with pytest.raises(ValueError):
        ab_test(visitors_a=0, conversions_a=0, visitors_b=100, conversions_b=10)

def test_ab_test_invalid_conversions():
    with pytest.raises(ValueError):
        ab_test(visitors_a=100, conversions_a=-1, visitors_b=100, conversions_b=10)

def test_ab_test_conversions_greater_than_visitors():
    with pytest.raises(ValueError):
        ab_test(visitors_a=100, conversions_a=200, visitors_b=100, conversions_b=10)

def test_bayesian_ab_test_basic():
    result = bayesian_ab_test(visitors_a=100, conversions_a=20, visitors_b=100, conversions_b=30)
    assert "prob_b_better" in result
    assert 0 <= result["prob_b_better"] <= 1

def test_bayesian_ab_test_more_conversions_in_a():
    result=bayesian_ab_test(visitors_a=100, conversions_a=50, visitors_b=100, conversions_b=40)
    assert "prob_b_better" in result
    assert result["prob_b_better"]<0.5

def test_bayesian_test_invalid_values():
    with pytest.raises(ValueError):
        bayesian_ab_test(visitors_a=0, conversions_a=0, visitors_b=100, conversions_b=20)
