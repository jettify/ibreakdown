from ibreakdown.iexplainer import features_groups


def test_features_pairs():
    result = features_groups(3)
    assert result == [0, 1, 2, (0, 1), (0, 2), (1, 2)]
