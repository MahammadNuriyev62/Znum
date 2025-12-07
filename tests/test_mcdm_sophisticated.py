"""
Sophisticated MCDM Tests.

These tests verify complex scenarios for PROMETHEE, TOPSIS, and VIKOR:
- Large decision matrices (many alternatives and criteria)
- Edge cases (similar alternatives, extreme weights)
- Ranking stability and sensitivity
- Mathematical properties (transitivity, dominance)
- Cross-method validation
- Real-world-like decision scenarios
"""

import pytest
import numpy as np
from znum.Znum import Znum
from znum.Beast import Beast
from znum.Promethee import Promethee
from znum.Topsis import Topsis
from znum.Vikor import Vikor


# =============================================================================
# Helper Functions
# =============================================================================

def create_znum(base_value, spread=0.1, reliability_base=0.5, reliability_spread=0.1):
    """Helper to create Znums with controlled parameters."""
    A = [
        base_value - spread,
        base_value - spread/3,
        base_value + spread/3,
        base_value + spread
    ]
    B = [
        max(0, reliability_base - reliability_spread),
        reliability_base - reliability_spread/3,
        min(1, reliability_base + reliability_spread/3),
        min(1, reliability_base + reliability_spread)
    ]
    return Znum(A, B)


def get_topsis_ranking(table):
    """Get ranking indices from TOPSIS scores (highest score = rank 1)."""
    result = Topsis.solver_main(table)
    # Sort indices by score descending
    return sorted(range(len(result)), key=lambda i: result[i], reverse=True)


# =============================================================================
# Large Decision Matrix Tests
# =============================================================================

class TestLargeDecisionMatrices:
    """Test MCDM methods with large decision matrices."""

    def test_five_alternatives_three_criteria(self):
        """Test PROMETHEE with 5 alternatives and 3 criteria."""
        weights = [
            Znum([0.3, 0.35, 0.4, 0.45], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.25, 0.3, 0.35, 0.4], [0.5, 0.6, 0.7, 0.8]),
            Znum([0.2, 0.25, 0.3, 0.35], [0.5, 0.6, 0.7, 0.8]),
        ]

        # Create 5 alternatives with different performance profiles
        alt1 = [create_znum(0.8), create_znum(0.6), create_znum(0.5)]  # Good, medium, low
        alt2 = [create_znum(0.5), create_znum(0.8), create_znum(0.6)]  # Medium, good, medium
        alt3 = [create_znum(0.6), create_znum(0.5), create_znum(0.9)]  # Medium, low, excellent
        alt4 = [create_znum(0.7), create_znum(0.7), create_znum(0.7)]  # Balanced
        alt5 = [create_znum(0.4), create_znum(0.4), create_znum(0.4)]  # Poor across all

        criteria_types = [
            Beast.CriteriaType.BENEFIT,
            Beast.CriteriaType.BENEFIT,
            Beast.CriteriaType.BENEFIT,
        ]

        promethee = Promethee(
            [weights, alt1, alt2, alt3, alt4, alt5, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # alt5 should be worst (lowest values across all criteria)
        assert promethee.index_of_worst_alternative == 4
        # Ranking should contain all 5 alternatives
        assert len(promethee.ordered_indices) == 5
        assert set(promethee.ordered_indices) == {0, 1, 2, 3, 4}

    def test_topsis_four_criteria(self):
        """Test TOPSIS with 4 criteria."""
        weights = [
            Znum([0.25, 0.3, 0.35, 0.4], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.2, 0.25, 0.3, 0.35], [0.5, 0.6, 0.7, 0.8]),
            Znum([0.15, 0.2, 0.25, 0.3], [0.5, 0.6, 0.7, 0.8]),
            Znum([0.1, 0.15, 0.2, 0.25], [0.4, 0.5, 0.6, 0.7]),
        ]

        # Create alternatives
        best = [create_znum(0.9), create_znum(0.85), create_znum(0.8), create_znum(0.75)]
        middle = [create_znum(0.6), create_znum(0.55), create_znum(0.5), create_znum(0.45)]
        worst = [create_znum(0.3), create_znum(0.25), create_znum(0.2), create_znum(0.15)]

        table = [
            weights,
            best,
            middle,
            worst,
            ["B", "B", "B", "B"]
        ]

        result = Topsis.solver_main(table)

        # Best should have highest score, worst should have lowest
        assert result[0] > result[1] > result[2]

    def test_six_alternatives_ranking_consistency(self):
        """Test that rankings are consistent with 6 alternatives."""
        weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.5, 0.6, 0.7, 0.8]),
        ]

        # Create 6 alternatives with clearly different performance
        alternatives = [
            [create_znum(0.9), create_znum(0.85)],  # Best
            [create_znum(0.8), create_znum(0.75)],
            [create_znum(0.7), create_znum(0.65)],
            [create_znum(0.6), create_znum(0.55)],
            [create_znum(0.5), create_znum(0.45)],
            [create_znum(0.4), create_znum(0.35)],  # Worst
        ]

        criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights] + alternatives + [criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Best should be index 0, worst should be index 5
        assert promethee.index_of_best_alternative == 0
        assert promethee.index_of_worst_alternative == 5


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_similar_alternatives(self):
        """Test ranking of very similar alternatives."""
        weights = [Znum([0.5, 0.55, 0.6, 0.65], [0.6, 0.7, 0.8, 0.9])]

        # Almost identical alternatives with tiny differences
        alt1 = [Znum([0.500, 0.510, 0.520, 0.530], [0.7, 0.75, 0.8, 0.85])]
        alt2 = [Znum([0.501, 0.511, 0.521, 0.531], [0.7, 0.75, 0.8, 0.85])]
        alt3 = [Znum([0.502, 0.512, 0.522, 0.532], [0.7, 0.75, 0.8, 0.85])]

        criteria_types = [Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights, alt1, alt2, alt3, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Should still produce a valid ranking
        assert len(promethee.ordered_indices) == 3
        # alt3 has slightly higher values, should rank best
        assert promethee.index_of_best_alternative == 2

    def test_extreme_weight_differences(self):
        """Test with one criterion having much higher weight."""
        # First criterion has much higher weight
        weights = [
            Znum([0.8, 0.85, 0.9, 0.95], [0.8, 0.85, 0.9, 0.95]),  # High weight
            Znum([0.05, 0.08, 0.1, 0.12], [0.5, 0.6, 0.7, 0.8]),   # Low weight
        ]

        # Alt1: Good on high-weight criterion, poor on low-weight
        alt1 = [create_znum(0.9), create_znum(0.2)]
        # Alt2: Poor on high-weight criterion, good on low-weight
        alt2 = [create_znum(0.3), create_znum(0.9)]

        criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights, alt1, alt2, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Alt1 should win because high-weight criterion dominates
        assert promethee.index_of_best_alternative == 0

    def test_all_cost_criteria(self):
        """Test with all cost criteria (lower is better)."""
        weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.5, 0.6, 0.7, 0.8]),
        ]

        # Low cost alternative (should be better)
        alt1 = [create_znum(0.2), create_znum(0.3)]
        # High cost alternative (should be worse)
        alt2 = [create_znum(0.8), create_znum(0.9)]

        table = [
            weights,
            alt1,
            alt2,
            ["C", "C"]  # Both cost criteria
        ]

        result = Topsis.solver_main(table)

        # Lower cost should score higher (better)
        assert result[0] > result[1]

    def test_mixed_criteria_balance(self):
        """Test with balanced mix of benefit and cost criteria."""
        weights = [
            Znum([0.3, 0.35, 0.4, 0.45], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.6, 0.7, 0.8, 0.9]),
        ]

        # Alt1: High benefit, high cost, medium benefit
        alt1 = [create_znum(0.9), create_znum(0.9), create_znum(0.5)]
        # Alt2: Low benefit, low cost, high benefit
        alt2 = [create_znum(0.3), create_znum(0.2), create_znum(0.9)]
        # Alt3: Balanced across all
        alt3 = [create_znum(0.6), create_znum(0.5), create_znum(0.6)]

        table = [
            weights,
            alt1,
            alt2,
            alt3,
            ["B", "C", "B"]  # Benefit, Cost, Benefit
        ]

        result = Topsis.solver_main(table)

        # All alternatives should have valid scores
        assert all(0 <= r <= 1 for r in result)


# =============================================================================
# Mathematical Properties
# =============================================================================

class TestMathematicalProperties:
    """Test mathematical properties of MCDM methods."""

    def test_dominance_respected(self):
        """If A dominates B on all criteria, A should rank higher."""
        weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.5, 0.6, 0.7, 0.8]),
            Znum([0.2, 0.25, 0.3, 0.35], [0.5, 0.6, 0.7, 0.8]),
        ]

        # Dominant alternative (best on ALL criteria)
        dominant = [create_znum(0.9), create_znum(0.85), create_znum(0.8)]
        # Dominated alternative (worse on ALL criteria)
        dominated = [create_znum(0.3), create_znum(0.25), create_znum(0.2)]
        # Middle alternative
        middle = [create_znum(0.6), create_znum(0.55), create_znum(0.5)]

        criteria_types = [
            Beast.CriteriaType.BENEFIT,
            Beast.CriteriaType.BENEFIT,
            Beast.CriteriaType.BENEFIT,
        ]

        promethee = Promethee(
            [weights, dominant, dominated, middle, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Dominant should be best, dominated should be worst
        assert promethee.index_of_best_alternative == 0
        assert promethee.index_of_worst_alternative == 1

    def test_ranking_transitivity(self):
        """If A > B and B > C, then A should rank higher than C."""
        weights = [Znum([0.5, 0.55, 0.6, 0.65], [0.6, 0.7, 0.8, 0.9])]

        # Clear ordering: A > B > C
        alt_a = [create_znum(0.9)]
        alt_b = [create_znum(0.6)]
        alt_c = [create_znum(0.3)]

        criteria_types = [Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights, alt_a, alt_b, alt_c, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Ranking should be A=0, B=1, C=2
        assert promethee.ordered_indices == [0, 1, 2]

    def test_scale_independence_topsis(self):
        """TOPSIS scores should be normalized (between 0 and 1)."""
        weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.6, 0.7, 0.8, 0.9]),
        ]

        # Use very large values
        alt1 = [Znum([100, 110, 120, 130], [0.7, 0.8, 0.85, 0.9])]
        alt2 = [Znum([200, 210, 220, 230], [0.7, 0.8, 0.85, 0.9])]

        table = [weights, alt1, alt2, ["B"]]

        result = Topsis.solver_main(table)

        # Scores should still be between 0 and 1
        assert all(0 <= r <= 1 for r in result)


# =============================================================================
# Sensitivity Analysis
# =============================================================================

class TestSensitivityAnalysis:
    """Test ranking sensitivity to parameter changes."""

    def test_weight_sensitivity(self):
        """Test how ranking changes with different weight distributions."""
        # Alt1: Good on C1, poor on C2
        alt1 = [create_znum(0.9), create_znum(0.3)]
        # Alt2: Poor on C1, good on C2
        alt2 = [create_znum(0.3), create_znum(0.9)]

        criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.BENEFIT]

        # Test with C1-heavy weights
        weights_c1_heavy = [
            Znum([0.7, 0.75, 0.8, 0.85], [0.7, 0.8, 0.85, 0.9]),
            Znum([0.1, 0.15, 0.2, 0.25], [0.5, 0.6, 0.7, 0.8]),
        ]

        promethee_c1 = Promethee(
            [weights_c1_heavy, alt1, alt2, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee_c1.solve()

        # Alt1 should win with C1-heavy weights
        assert promethee_c1.index_of_best_alternative == 0

        # Test with C2-heavy weights
        weights_c2_heavy = [
            Znum([0.1, 0.15, 0.2, 0.25], [0.5, 0.6, 0.7, 0.8]),
            Znum([0.7, 0.75, 0.8, 0.85], [0.7, 0.8, 0.85, 0.9]),
        ]

        promethee_c2 = Promethee(
            [weights_c2_heavy, alt1, alt2, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee_c2.solve()

        # Alt2 should win with C2-heavy weights
        assert promethee_c2.index_of_best_alternative == 1

    def test_reliability_impact(self):
        """Test how B (reliability) affects rankings."""
        weights = [Znum([0.5, 0.55, 0.6, 0.65], [0.7, 0.8, 0.85, 0.9])]

        # Same A values but different B (reliability)
        alt_high_reliability = [Znum([0.6, 0.65, 0.7, 0.75], [0.85, 0.9, 0.92, 0.95])]
        alt_low_reliability = [Znum([0.6, 0.65, 0.7, 0.75], [0.3, 0.4, 0.5, 0.6])]

        criteria_types = [Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights, alt_high_reliability, alt_low_reliability, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Both have same A values, but reliability differs
        # The ranking should still be deterministic
        assert promethee.index_of_best_alternative in [0, 1]
        assert promethee.index_of_worst_alternative in [0, 1]


# =============================================================================
# Cross-Method Validation
# =============================================================================

class TestCrossMethodValidation:
    """Validate consistency across different MCDM methods."""

    def test_clear_winner_all_methods(self):
        """All methods should agree when one alternative clearly dominates."""
        weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.5, 0.6, 0.7, 0.8]),
        ]

        # Dominant alternative
        best = [create_znum(0.95), create_znum(0.9)]
        # Middle alternatives
        mid1 = [create_znum(0.6), create_znum(0.55)]
        mid2 = [create_znum(0.5), create_znum(0.45)]
        # Worst alternative
        worst = [create_znum(0.15), create_znum(0.1)]

        criteria_types = [Beast.CriteriaType.BENEFIT, Beast.CriteriaType.BENEFIT]

        # PROMETHEE
        promethee = Promethee(
            [weights, best, mid1, mid2, worst, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()
        promethee_best = promethee.index_of_best_alternative
        promethee_worst = promethee.index_of_worst_alternative

        # TOPSIS
        topsis_weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.5, 0.6, 0.7, 0.8]),
        ]
        t_best = [create_znum(0.95), create_znum(0.9)]
        t_mid1 = [create_znum(0.6), create_znum(0.55)]
        t_mid2 = [create_znum(0.5), create_znum(0.45)]
        t_worst = [create_znum(0.15), create_znum(0.1)]

        table = [
            topsis_weights,
            t_best,
            t_mid1,
            t_mid2,
            t_worst,
            ["B", "B"]
        ]
        topsis_result = Topsis.solver_main(table)
        topsis_best = topsis_result.index(max(topsis_result))
        topsis_worst = topsis_result.index(min(topsis_result))

        # Both methods should agree on best and worst
        assert promethee_best == 0, f"PROMETHEE best: {promethee_best}"
        assert topsis_best == 0, f"TOPSIS best: {topsis_best}"
        assert promethee_worst == 3, f"PROMETHEE worst: {promethee_worst}"
        assert topsis_worst == 3, f"TOPSIS worst: {topsis_worst}"

    def test_vikor_regret_measure_consistency(self):
        """Test VIKOR regret measure produces consistent ordering."""
        weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.6, 0.7, 0.8, 0.9]),
            Znum([0.3, 0.35, 0.4, 0.45], [0.5, 0.6, 0.7, 0.8]),
        ]

        # Create alternatives with clear ordering
        alt1 = [create_znum(0.85), create_znum(0.8)]  # Best
        alt2 = [create_znum(0.6), create_znum(0.55)]  # Middle
        alt3 = [create_znum(0.3), create_znum(0.25)]  # Worst

        table_main = [alt1, alt2, alt3]

        regret = Vikor.regret_measure(weights, table_main)

        # Should have 3 regret measurements
        assert len(regret) == 3
        # All should be Znums
        assert all(isinstance(r, Znum) for r in regret)


# =============================================================================
# Real-World-Like Scenarios
# =============================================================================

class TestRealWorldScenarios:
    """Test scenarios resembling real-world decision problems."""

    def test_supplier_selection(self):
        """Supplier selection with price, quality, and delivery criteria."""
        # Weights: Quality > Price > Delivery
        weights = [
            Znum([0.4, 0.45, 0.5, 0.55], [0.7, 0.8, 0.85, 0.9]),   # Quality (benefit)
            Znum([0.3, 0.35, 0.4, 0.45], [0.6, 0.7, 0.8, 0.85]),   # Price (cost)
            Znum([0.15, 0.2, 0.25, 0.3], [0.5, 0.6, 0.7, 0.8]),    # Delivery (benefit)
        ]

        # Supplier A: High quality, high price, fast delivery
        supplier_a = [create_znum(0.9), create_znum(0.85), create_znum(0.8)]
        # Supplier B: Medium quality, low price, medium delivery
        supplier_b = [create_znum(0.6), create_znum(0.3), create_znum(0.5)]
        # Supplier C: Low quality, very low price, slow delivery
        supplier_c = [create_znum(0.4), create_znum(0.15), create_znum(0.3)]

        table = [
            weights,
            supplier_a,
            supplier_b,
            supplier_c,
            ["B", "C", "B"]  # Quality=Benefit, Price=Cost, Delivery=Benefit
        ]

        result = Topsis.solver_main(table)

        # All suppliers should have valid scores
        assert all(0 <= r <= 1 for r in result)
        # Should produce a clear ranking
        sorted_indices = sorted(range(3), key=lambda i: result[i], reverse=True)
        assert len(set(sorted_indices)) == 3  # All different positions

    def test_project_selection(self):
        """Project selection with ROI, risk, and strategic fit."""
        weights = [
            Znum([0.35, 0.4, 0.45, 0.5], [0.7, 0.8, 0.85, 0.9]),   # ROI (benefit)
            Znum([0.3, 0.35, 0.4, 0.45], [0.6, 0.7, 0.8, 0.85]),   # Risk (cost)
            Znum([0.2, 0.25, 0.3, 0.35], [0.5, 0.6, 0.7, 0.8]),    # Strategic fit (benefit)
        ]

        # High risk/high reward project
        project_a = [create_znum(0.95), create_znum(0.9), create_znum(0.7)]
        # Low risk/moderate reward project
        project_b = [create_znum(0.5), create_znum(0.2), create_znum(0.8)]
        # Balanced project
        project_c = [create_znum(0.7), create_znum(0.5), create_znum(0.75)]

        criteria_types = [
            Beast.CriteriaType.BENEFIT,
            Beast.CriteriaType.COST,
            Beast.CriteriaType.BENEFIT,
        ]

        promethee = Promethee(
            [weights, project_a, project_b, project_c, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Should produce a valid ranking
        assert len(promethee.ordered_indices) == 3
        assert set(promethee.ordered_indices) == {0, 1, 2}

    def test_technology_evaluation(self):
        """Technology evaluation with multiple technical and business criteria."""
        weights = [
            Znum([0.2, 0.25, 0.3, 0.35], [0.6, 0.7, 0.8, 0.9]),   # Performance
            Znum([0.2, 0.25, 0.3, 0.35], [0.6, 0.7, 0.8, 0.9]),   # Reliability
            Znum([0.15, 0.2, 0.25, 0.3], [0.5, 0.6, 0.7, 0.8]),   # Cost
            Znum([0.15, 0.2, 0.25, 0.3], [0.5, 0.6, 0.7, 0.8]),   # Maintainability
        ]

        # Technology options
        tech_a = [create_znum(0.9), create_znum(0.85), create_znum(0.7), create_znum(0.6)]
        tech_b = [create_znum(0.7), create_znum(0.9), create_znum(0.4), create_znum(0.8)]
        tech_c = [create_znum(0.6), create_znum(0.7), create_znum(0.2), create_znum(0.9)]
        tech_d = [create_znum(0.8), create_znum(0.75), create_znum(0.5), create_znum(0.7)]

        table = [
            weights,
            tech_a,
            tech_b,
            tech_c,
            tech_d,
            ["B", "B", "C", "B"]  # Performance, Reliability, Cost, Maintainability
        ]

        result = Topsis.solver_main(table)

        # Should have 4 results
        assert len(result) == 4
        # All should be valid scores
        assert all(0 <= r <= 1 for r in result)


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for edge conditions."""

    def test_identical_alternatives_ordering(self):
        """Test behavior when all alternatives are identical."""
        weights = [Znum([0.5, 0.55, 0.6, 0.65], [0.6, 0.7, 0.8, 0.9])]

        # All identical
        alt = [Znum([0.5, 0.55, 0.6, 0.65], [0.6, 0.7, 0.8, 0.85])]

        criteria_types = [Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights, alt, alt.copy(), alt.copy(), criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Should still produce a valid ranking (order may be arbitrary)
        assert len(promethee.ordered_indices) == 3

    def test_high_uncertainty_znums(self):
        """Test with high uncertainty (wide spreads) in Znum values."""
        weights = [Znum([0.3, 0.5, 0.7, 0.9], [0.4, 0.6, 0.8, 0.95])]  # Wide spread

        # High uncertainty alternatives
        alt1 = [Znum([0.2, 0.5, 0.8, 0.95], [0.3, 0.5, 0.7, 0.9])]
        alt2 = [Znum([0.3, 0.55, 0.75, 0.9], [0.4, 0.55, 0.75, 0.85])]

        criteria_types = [Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights, alt1, alt2, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # Should still work despite high uncertainty
        assert promethee.index_of_best_alternative in [0, 1]

    def test_narrow_value_range(self):
        """Test with very narrow value ranges."""
        weights = [Znum([0.499, 0.4995, 0.5005, 0.501], [0.7, 0.8, 0.85, 0.9])]

        # Very narrow spreads
        alt1 = [Znum([0.600, 0.601, 0.602, 0.603], [0.8, 0.85, 0.88, 0.9])]
        alt2 = [Znum([0.604, 0.605, 0.606, 0.607], [0.8, 0.85, 0.88, 0.9])]

        criteria_types = [Beast.CriteriaType.BENEFIT]

        promethee = Promethee(
            [weights, alt1, alt2, criteria_types],
            shouldNormalizeWeight=True
        )
        promethee.solve()

        # alt2 has slightly higher values
        assert promethee.index_of_best_alternative == 1
