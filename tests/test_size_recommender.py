"""Unit tests for size recommendation engine."""

import pytest
from src.size_recommender import SizeRecommender

@pytest.fixture
def recommender():
    return SizeRecommender("config/size_chart.json")

def test_chart_loading(recommender):
    assert len(recommender.sizes) >= 7
    assert recommender.metadata.get("name") is not None

def test_exact_size_recommendations(recommender):
    # Test Size 6 (around 244mm)
    res6 = recommender.recommend_size(244.0)
    assert res6.is_covered is True
    assert res6.recommended_size == 6

    # Test Size 8 (around 261mm)
    res8 = recommender.recommend_size(261.0)
    assert res8.is_covered is True
    assert res8.recommended_size == 8
    assert res8.size_eu == 41

    # Test Size 10 (around 279mm)
    res10 = recommender.recommend_size(279.0)
    assert res10.is_covered is True
    assert res10.recommended_size == 10

def test_boundary_cases(recommender):
    # Size 8 max is 266.0, Size 9 min is 266.1
    # 265.8 should trigger boundary warning near size 9 or 8
    res = recommender.recommend_size(265.5)
    assert res.recommended_size == 8
    assert res.boundary_warning is not None
    assert "Size 9" in res.boundary_warning

def test_out_of_range_handling(recommender):
    # Too small (< 240mm)
    res_small = recommender.recommend_size(200.0)
    assert res_small.is_covered is False
    assert res_small.recommended_size is None
    assert "Below Minimum" in res_small.label

    # Too large (> 305mm)
    res_large = recommender.recommend_size(350.0)
    assert res_large.is_covered is False
    assert res_large.recommended_size is None
    assert "Above Maximum" in res_large.label
