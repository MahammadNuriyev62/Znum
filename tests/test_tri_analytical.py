"""Tests for Li et al. 2023 analytical engine for triangular Z-numbers.

Tests the extended triangular PDF, analytical matrix, convolution,
A computation, B computation, and full tri_solve pipeline.
"""
import numpy as np
import pytest
from znum import Znum
from znum.math_ops import Math
from znum.tri_math import (
    make_ext_tri_params,
    eval_ext_tri_pdf,
    analytical_matrix,
    compute_A_tri,
    convolve_pdfs,
    compute_base_value,
    compute_B_tri,
    tri_solve,
    _tri_membership,
)


# ======================================================================
# Phase 1: Extended Triangular PDF
# ======================================================================

class TestExtTriParams:
    """Test make_ext_tri_params() — Eq. 36."""

    def test_v1_peaked(self):
        """v=1: beta=0, h=2/w — peaked distribution."""
        beta, h = make_ext_tri_params(1, 2, 3, v=1.0)
        assert beta == pytest.approx(0.0)
        assert h == pytest.approx(1.0)  # 2/(3-1) = 1

    def test_v0_bathtub(self):
        """v=0: beta=2/w, h=0 — bathtub distribution."""
        beta, h = make_ext_tri_params(1, 2, 3, v=0.0)
        assert beta == pytest.approx(1.0)  # 2/(3-1) = 1
        assert h == pytest.approx(0.0)

    def test_v05_uniform(self):
        """v=0.5: beta=h=1/w — uniform distribution."""
        beta, h = make_ext_tri_params(1, 2, 3, v=0.5)
        assert beta == pytest.approx(0.5)  # 1/(3-1) = 0.5
        assert h == pytest.approx(0.5)

    def test_asymmetric_triangle(self):
        """Asymmetric triangle (a=0, m=1, b=5)."""
        beta, h = make_ext_tri_params(0, 1, 5, v=0.8)
        assert beta == pytest.approx(2 * 0.2 / 5)
        assert h == pytest.approx(2 * 0.8 / 5)

    def test_degenerate_raises(self):
        """a==b should raise ValueError."""
        with pytest.raises(ValueError, match="Degenerate"):
            make_ext_tri_params(3, 3, 3, v=0.5)


class TestExtTriPDF:
    """Test eval_ext_tri_pdf() — Eq. 32."""

    def test_integrates_to_1_v1(self):
        """PDF with v=1 integrates to 1."""
        beta, h = make_ext_tri_params(1, 2, 3, v=1.0)
        x = np.linspace(0, 4, 10000)
        pdf = eval_ext_tri_pdf(1, 2, 3, beta, h, x)
        integral = np.trapz(pdf, x)
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_integrates_to_1_v0(self):
        """PDF with v=0 integrates to 1."""
        beta, h = make_ext_tri_params(1, 2, 3, v=0.0)
        x = np.linspace(0, 4, 10000)
        pdf = eval_ext_tri_pdf(1, 2, 3, beta, h, x)
        integral = np.trapz(pdf, x)
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_integrates_to_1_v05(self):
        """PDF with v=0.5 integrates to 1."""
        beta, h = make_ext_tri_params(1, 2, 3, v=0.5)
        x = np.linspace(0, 4, 10000)
        pdf = eval_ext_tri_pdf(1, 2, 3, beta, h, x)
        integral = np.trapz(pdf, x)
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_integrates_to_1_various_v(self):
        """PDF integrates to 1 for various v values."""
        for v in [0.1, 0.3, 0.7, 0.9]:
            beta, h = make_ext_tri_params(2, 5, 8, v)
            x = np.linspace(1, 9, 10000)
            pdf = eval_ext_tri_pdf(2, 5, 8, beta, h, x)
            integral = np.trapz(pdf, x)
            assert integral == pytest.approx(1.0, abs=1e-3), f"Failed for v={v}"

    def test_zero_outside_support(self):
        """PDF is zero outside [a, b]."""
        beta, h = make_ext_tri_params(1, 2, 3, v=0.7)
        x = np.array([0.0, 0.5, 0.999, 3.001, 4.0, 5.0])
        pdf = eval_ext_tri_pdf(1, 2, 3, beta, h, x)
        np.testing.assert_array_equal(pdf, 0.0)

    def test_peaked_at_m_when_v1(self):
        """When v=1, PDF peaks at m and is zero at endpoints."""
        beta, h = make_ext_tri_params(1, 2, 3, v=1.0)
        pdf_a = eval_ext_tri_pdf(1, 2, 3, beta, h, np.array([1.0]))
        pdf_m = eval_ext_tri_pdf(1, 2, 3, beta, h, np.array([2.0]))
        pdf_b = eval_ext_tri_pdf(1, 2, 3, beta, h, np.array([3.0]))
        assert pdf_a[0] == pytest.approx(0.0)
        assert pdf_b[0] == pytest.approx(0.0)
        assert pdf_m[0] > 0

    def test_bathtub_at_v0(self):
        """When v=0, PDF is highest at endpoints, zero at peak."""
        beta, h = make_ext_tri_params(1, 2, 3, v=0.0)
        pdf_a = eval_ext_tri_pdf(1, 2, 3, beta, h, np.array([1.0]))
        pdf_m = eval_ext_tri_pdf(1, 2, 3, beta, h, np.array([2.0]))
        pdf_b = eval_ext_tri_pdf(1, 2, 3, beta, h, np.array([3.0]))
        assert pdf_m[0] == pytest.approx(0.0)
        assert pdf_a[0] > 0
        assert pdf_b[0] > 0

    def test_asymmetric_triangle(self):
        """Asymmetric triangle: f(m) = h, f(a) = f(b) = beta."""
        beta, h = make_ext_tri_params(0, 1, 5, v=0.7)
        pdf = eval_ext_tri_pdf(0, 1, 5, beta, h, np.array([0.0, 1.0, 5.0]))
        assert pdf[0] == pytest.approx(beta)
        assert pdf[1] == pytest.approx(h)
        assert pdf[2] == pytest.approx(beta)


# ======================================================================
# Phase 2: Analytical Matrix
# ======================================================================

class TestAnalyticalMatrix:
    """Test analytical_matrix() — drop-in for get_matrix()."""

    def test_shape_matches_lp(self):
        """Analytical matrix has same shape as LP matrix."""
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        lp_matrix = z.math._get_matrix_lp()
        an_matrix = analytical_matrix(z)
        assert an_matrix.shape == lp_matrix.shape

    def test_columns_sum_to_1(self):
        """Each column is a probability distribution (sums to 1)."""
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        matrix = analytical_matrix(z)
        for j in range(matrix.shape[1]):
            assert matrix[:, j].sum() == pytest.approx(1.0, abs=1e-10)

    def test_all_nonnegative(self):
        """All values are non-negative."""
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        matrix = analytical_matrix(z)
        assert np.all(matrix >= 0)

    def test_works_in_fast_mode(self):
        """get_matrix() dispatches to analytical in fast mode."""
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        with Znum.fast():
            matrix = z.math.get_matrix()
        # Should be analytical — verify shape
        assert matrix.shape == (10, 10)
        for j in range(matrix.shape[1]):
            assert matrix[:, j].sum() == pytest.approx(1.0, abs=1e-10)

    def test_lp_without_fast(self):
        """get_matrix() still uses LP without fast mode."""
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        matrix1 = z.math.get_matrix()  # LP
        matrix2 = z.math._get_matrix_lp()  # explicit LP
        np.testing.assert_array_almost_equal(matrix1, matrix2)

    def test_low_reliability(self):
        """Works with low reliability values."""
        z = Znum([1, 2, 2, 3], [0.1, 0.2, 0.2, 0.3])
        matrix = analytical_matrix(z)
        assert matrix.shape == (10, 10)
        for j in range(matrix.shape[1]):
            assert matrix[:, j].sum() == pytest.approx(1.0, abs=1e-10)

    def test_high_reliability(self):
        """Works with high reliability near 1."""
        z = Znum([1, 2, 2, 3], [0.9, 0.95, 0.95, 1.0])
        matrix = analytical_matrix(z)
        for j in range(matrix.shape[1]):
            assert matrix[:, j].sum() == pytest.approx(1.0, abs=1e-10)


# ======================================================================
# Phase 2b: Hellinger distance with analytical matrix
# ======================================================================

class TestHellingerWithAnalytical:
    """Hellinger distance must work with analytical matrix."""

    def test_hellinger_runs_in_fast(self):
        """Hellinger distance computes without error in fast mode."""
        from znum.dist import Dist
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([4, 5, 5, 6], [0.6, 0.7, 0.7, 0.8])
        with Znum.fast():
            d = Dist.Hellinger.calculate(z1, z2)
        assert isinstance(d, float)
        assert d >= 0

    def test_hellinger_identical_znums(self):
        """Distance of identical Z-numbers should be near 0."""
        from znum.dist import Dist
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        with Znum.fast():
            d = Dist.Hellinger.calculate(z1, z1)
        assert d == pytest.approx(0.0, abs=0.05)


# ======================================================================
# Phase 3: A computation
# ======================================================================

class TestComputeATri:
    """Test compute_A_tri() — closed-form A for triangular."""

    def test_addition(self):
        A1 = np.array([1.0, 2.0, 3.0])
        A2 = np.array([4.0, 6.0, 8.0])
        result = compute_A_tri(A1, A2, Math.Operations.ADDITION)
        np.testing.assert_array_almost_equal(result, [5, 8, 11])

    def test_subtraction(self):
        A1 = np.array([4.0, 6.0, 8.0])
        A2 = np.array([1.0, 2.0, 3.0])
        result = compute_A_tri(A1, A2, Math.Operations.SUBTRACTION)
        np.testing.assert_array_almost_equal(result, [1, 4, 7])

    def test_multiplication_positive(self):
        A1 = np.array([1.0, 2.0, 3.0])
        A2 = np.array([4.0, 5.0, 6.0])
        result = compute_A_tri(A1, A2, Math.Operations.MULTIPLICATION)
        assert result[0] == pytest.approx(4.0)   # min(1*4, 1*6, 3*4, 3*6)
        assert result[1] == pytest.approx(10.0)  # 2*5
        assert result[2] == pytest.approx(18.0)  # max corners

    def test_division_positive(self):
        A1 = np.array([2.0, 4.0, 6.0])
        A2 = np.array([1.0, 2.0, 3.0])
        result = compute_A_tri(A1, A2, Math.Operations.DIVISION)
        assert result[0] == pytest.approx(2.0 / 3.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(6.0)

    def test_division_by_zero_raises(self):
        A1 = np.array([1.0, 2.0, 3.0])
        A2 = np.array([-1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="Division by zero"):
            compute_A_tri(A1, A2, Math.Operations.DIVISION)

    def test_addition_matches_znum_a(self):
        """Analytical A matches Znum's cross-product A for addition."""
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([4, 6, 6, 8], [0.6, 0.7, 0.7, 0.8])
        # Znum A result via cross-product
        a_pairs = Math._compute_a_pairs(z1, z2, Math.Operations.ADDITION)
        merged = Math._merge_rows(a_pairs)
        A_lp = Math._extract_trapezoid(merged)
        # Analytical A
        A_tri = compute_A_tri(z1.A_tri, z2.A_tri, Math.Operations.ADDITION)
        # Compare: A_lp is [a,b,b,c], A_tri is (a,b,c)
        assert A_tri[0] == pytest.approx(A_lp[0], abs=1e-4)
        assert A_tri[1] == pytest.approx(A_lp[1], abs=1e-4)
        assert A_tri[2] == pytest.approx(A_lp[3], abs=1e-4)


# ======================================================================
# Phase 3: Convolution
# ======================================================================

class TestConvolution:
    """Test convolve_pdfs() — numerical convolution."""

    def test_addition_support(self):
        """Addition convolution has support [a1+a2, b1+b2]."""
        z_grid, f = convolve_pdfs(1, 2, 3, 0.8, 4, 6, 8, 0.7,
                                  Math.Operations.ADDITION)
        assert z_grid[0] == pytest.approx(5.0)   # 1+4
        assert z_grid[-1] == pytest.approx(11.0)  # 3+8

    def test_addition_integrates_near_1(self):
        """Addition convolution integrates to ~1."""
        z_grid, f = convolve_pdfs(1, 2, 3, 0.8, 4, 6, 8, 0.7,
                                  Math.Operations.ADDITION)
        dz = z_grid[1] - z_grid[0]
        integral = np.trapz(f, dx=dz)
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_subtraction_support(self):
        """Subtraction convolution has support [a1-b2, b1-a2]."""
        z_grid, f = convolve_pdfs(4, 6, 8, 0.8, 1, 2, 3, 0.7,
                                  Math.Operations.SUBTRACTION)
        assert z_grid[0] == pytest.approx(1.0)   # 4-3
        assert z_grid[-1] == pytest.approx(7.0)  # 8-1

    def test_subtraction_integrates_near_1(self):
        z_grid, f = convolve_pdfs(4, 6, 8, 0.8, 1, 2, 3, 0.7,
                                  Math.Operations.SUBTRACTION)
        dz = z_grid[1] - z_grid[0]
        integral = np.trapz(f, dx=dz)
        assert integral == pytest.approx(1.0, abs=0.05)

    def test_multiplication_integrates_near_1(self):
        z_grid, f = convolve_pdfs(1, 2, 3, 0.8, 4, 5, 6, 0.7,
                                  Math.Operations.MULTIPLICATION)
        dz = z_grid[1] - z_grid[0]
        integral = np.trapz(f, dx=dz)
        assert integral == pytest.approx(1.0, abs=0.1)

    def test_division_integrates_near_1(self):
        z_grid, f = convolve_pdfs(2, 4, 6, 0.8, 1, 2, 3, 0.7,
                                  Math.Operations.DIVISION)
        dz = z_grid[1] - z_grid[0]
        integral = np.trapz(f, dx=dz)
        assert integral == pytest.approx(1.0, abs=0.1)

    def test_nonnegative(self):
        """Convolution result is non-negative."""
        z_grid, f = convolve_pdfs(1, 2, 3, 0.8, 4, 6, 8, 0.7,
                                  Math.Operations.ADDITION)
        assert np.all(f >= -1e-10)


# ======================================================================
# Phase 4: B computation
# ======================================================================

class TestComputeBTri:
    """Test compute_B_tri() and compute_base_value()."""

    def test_base_value_in_01(self):
        """Base value is in [0, 1]."""
        A_result = np.array([5.0, 8.0, 11.0])
        z_grid, f = convolve_pdfs(1, 2, 3, 0.8, 4, 6, 8, 0.7,
                                  Math.Operations.ADDITION)
        bv = compute_base_value(A_result, z_grid, f)
        assert 0 <= bv <= 1

    def test_base_value_higher_for_high_v(self):
        """Higher v (more peaked PDF) → higher base value."""
        A_result = np.array([5.0, 8.0, 11.0])
        _, f_low = convolve_pdfs(1, 2, 3, 0.3, 4, 6, 8, 0.3,
                                 Math.Operations.ADDITION)
        z_low, _ = convolve_pdfs(1, 2, 3, 0.3, 4, 6, 8, 0.3,
                                 Math.Operations.ADDITION)
        z_high, f_high = convolve_pdfs(1, 2, 3, 0.9, 4, 6, 8, 0.9,
                                       Math.Operations.ADDITION)
        bv_low = compute_base_value(A_result, z_low, f_low)
        bv_high = compute_base_value(A_result, z_high, f_high)
        assert bv_high > bv_low

    def test_b_result_monotonic(self):
        """B result is monotonically non-decreasing [b_L <= b_M <= b_R]."""
        A1 = np.array([1.0, 2.0, 3.0])
        B1 = np.array([0.3, 0.4, 0.5])
        A2 = np.array([4.0, 6.0, 8.0])
        B2 = np.array([0.6, 0.7, 0.8])
        A_r = compute_A_tri(A1, A2, Math.Operations.ADDITION)
        B_r = compute_B_tri(A1, B1, A2, B2, A_r, Math.Operations.ADDITION)
        assert B_r[0] <= B_r[1]
        assert B_r[1] <= B_r[2]

    def test_b_result_in_01(self):
        """All B result values are in [0, 1]."""
        A1 = np.array([1.0, 2.0, 3.0])
        B1 = np.array([0.3, 0.4, 0.5])
        A2 = np.array([4.0, 6.0, 8.0])
        B2 = np.array([0.6, 0.7, 0.8])
        A_r = compute_A_tri(A1, A2, Math.Operations.ADDITION)
        B_r = compute_B_tri(A1, B1, A2, B2, A_r, Math.Operations.ADDITION)
        assert np.all(B_r >= 0)
        assert np.all(B_r <= 1)

    def test_paper_example4(self):
        """Verify against Li et al. 2023, Example 4 (addition).

        Z1 = ((1,2,3), (0.7,0.8,0.9)), Z2 = ((7,8,9), (0.4,0.5,0.6))
        Paper gives B12 support [417/619, 711/1000], peak at 83/120.
        """
        A1 = np.array([1.0, 2.0, 3.0])
        B1 = np.array([0.7, 0.8, 0.9])
        A2 = np.array([7.0, 8.0, 9.0])
        B2 = np.array([0.4, 0.5, 0.6])
        A_r = compute_A_tri(A1, A2, Math.Operations.ADDITION)
        B_r = compute_B_tri(A1, B1, A2, B2, A_r, Math.Operations.ADDITION)
        # Closed-form: v12 = (v1+v2)/20 + (v1*v2)/15 + 3/5
        expected_L = (0.7 + 0.4) / 20 + (0.7 * 0.4) / 15 + 3 / 5  # 417/619
        expected_M = (0.8 + 0.5) / 20 + (0.8 * 0.5) / 15 + 3 / 5  # 83/120
        expected_R = (0.9 + 0.6) / 20 + (0.9 * 0.6) / 15 + 3 / 5  # 711/1000
        assert B_r[0] == pytest.approx(expected_L, abs=0.005)
        assert B_r[1] == pytest.approx(expected_M, abs=0.005)
        assert B_r[2] == pytest.approx(expected_R, abs=0.005)

    def test_high_b_does_not_inflate(self):
        """For high-reliability inputs, B result stays below max input B."""
        A1 = np.array([1.0, 2.0, 3.0])
        B1 = np.array([0.7, 0.8, 0.9])
        A2 = np.array([1.0, 2.0, 3.0])
        B2 = np.array([0.7, 0.8, 0.9])
        A_r = compute_A_tri(A1, A2, Math.Operations.ADDITION)
        B_r = compute_B_tri(A1, B1, A2, B2, A_r, Math.Operations.ADDITION)
        assert B_r.max() <= 0.9, (
            f"B inflated: result max {B_r.max():.4f} > input max 0.9"
        )


# ======================================================================
# Phase 5: Full tri_solve
# ======================================================================

class TestTriSolve:
    """Test tri_solve() — full analytical pipeline."""

    def test_addition_basic(self):
        """Basic addition returns valid Znum."""
        z1 = Znum([1, 2, 2, 3], [0.3, 0.4, 0.4, 0.5])
        z2 = Znum([4, 6, 6, 8], [0.6, 0.7, 0.7, 0.8])
        result = tri_solve(z1, z2, Math.Operations.ADDITION)
        assert isinstance(result, Znum)
        # A result: [5, 8, 8, 11]
        assert result.A[0] == pytest.approx(5.0)
        assert result.A[1] == pytest.approx(8.0)
        assert result.A[3] == pytest.approx(11.0)
        # B valid
        assert all(0 <= b <= 1 for b in result.B)

    def test_subtraction_basic(self):
        z1 = Znum([4, 6, 6, 8], [0.6, 0.7, 0.7, 0.8])
        z2 = Znum([1, 2, 2, 3], [0.3, 0.4, 0.4, 0.5])
        result = tri_solve(z1, z2, Math.Operations.SUBTRACTION)
        # A: [4-3, 6-2, 6-2, 8-1] = [1, 4, 4, 7]
        assert result.A[0] == pytest.approx(1.0)
        assert result.A[1] == pytest.approx(4.0)
        assert result.A[3] == pytest.approx(7.0)

    def test_multiplication_basic(self):
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([4, 5, 5, 6], [0.6, 0.7, 0.7, 0.8])
        result = tri_solve(z1, z2, Math.Operations.MULTIPLICATION)
        assert result.A[0] == pytest.approx(4.0)
        assert result.A[1] == pytest.approx(10.0)
        assert result.A[3] == pytest.approx(18.0)

    def test_division_basic(self):
        z1 = Znum([2, 4, 4, 6], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([1, 2, 2, 3], [0.6, 0.7, 0.7, 0.8])
        result = tri_solve(z1, z2, Math.Operations.DIVISION)
        assert result.A[0] == pytest.approx(2.0 / 3.0, abs=0.01)
        assert result.A[1] == pytest.approx(2.0, abs=0.01)
        assert result.A[3] == pytest.approx(6.0, abs=0.01)


# ======================================================================
# Phase 5: Operator integration (Znum.fast())
# ======================================================================

class TestFastModeOperators:
    """Test that Znum.fast() correctly dispatches to analytical path."""

    def test_addition_in_fast(self):
        """z1 + z2 in fast mode returns valid Znum."""
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([4, 5, 5, 6], [0.6, 0.7, 0.7, 0.8])
        with Znum.fast():
            r = z1 + z2
        assert r.A[0] == pytest.approx(5.0)
        assert r.A[1] == pytest.approx(7.0)
        assert r.A[3] == pytest.approx(9.0)
        assert all(0 <= b <= 1 for b in r.B)

    def test_subtraction_in_fast(self):
        z1 = Znum([4, 6, 6, 8], [0.6, 0.7, 0.7, 0.8])
        z2 = Znum([1, 2, 2, 3], [0.3, 0.4, 0.4, 0.5])
        with Znum.fast():
            r = z1 - z2
        assert r.A[0] == pytest.approx(1.0)
        assert r.A[3] == pytest.approx(7.0)

    def test_multiplication_in_fast(self):
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([4, 5, 5, 6], [0.6, 0.7, 0.7, 0.8])
        with Znum.fast():
            r = z1 * z2
        assert r.A[0] == pytest.approx(4.0)
        assert r.A[1] == pytest.approx(10.0)

    def test_division_in_fast(self):
        z1 = Znum([2, 4, 4, 6], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([1, 2, 2, 3], [0.6, 0.7, 0.7, 0.8])
        with Znum.fast():
            r = z1 / z2
        assert r.A[1] == pytest.approx(2.0)

    def test_non_triangular_falls_back_to_lp(self):
        """Non-triangular Z-numbers use LP even in fast mode."""
        z1 = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        z2 = Znum([2, 4, 8, 10], [0.5, 0.6, 0.7, 0.8])
        with Znum.fast():
            r = z1 + z2
        # Should still work (LP fallback)
        assert isinstance(r, Znum)
        assert r.A[0] == pytest.approx(3.0)

    def test_without_fast_uses_lp(self):
        """Without fast mode, triangular Z-numbers still use LP."""
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([4, 5, 5, 6], [0.6, 0.7, 0.7, 0.8])
        r_default = z1 + z2  # LP
        # A should be the same regardless of method
        assert r_default.A[0] == pytest.approx(5.0)


# ======================================================================
# Phase 6: B inflation bug fix
# ======================================================================

class TestBBehavior:
    """Tests for B behavior under Li et al. 2023 analytical method.

    Key properties:
    - B degrades through chained operations (high-B inputs)
    - Analytical gives narrower B spread than LP (high-B inputs)
    - For low-B inputs, B inflation is inherent (base value ≈ 0.6+)
      because Prob(X₁₂ ∈ A₁₂) is naturally high when A₁₂ support
      matches the convolution support.
    """

    def test_high_b_no_inflate(self):
        """High-reliability inputs: B stays below max input B."""
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        with Znum.fast():
            r = z1 + z2
        assert max(r.B) <= 0.9 + 0.02, (
            f"B inflated! Result B={r.B.tolist()}, max input B=0.9"
        )

    def test_chained_b_degrades(self):
        """B degrades monotonically through chained additions."""
        z = Znum([1, 2, 2, 3], [0.6, 0.7, 0.7, 0.8])
        with Znum.fast():
            r = z + z
            r2 = r + z
            r3 = r2 + z
        # Each chain step must have same or lower B peak
        assert max(r.B) <= max(z.B) + 0.02
        assert max(r2.B) <= max(r.B) + 0.02
        assert max(r3.B) <= max(r2.B) + 0.02

    def test_analytical_narrower_spread_than_lp(self):
        """Analytical B has narrower spread than LP for high-B inputs."""
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([4, 5, 5, 6], [0.6, 0.7, 0.7, 0.8])
        r_lp = z1 + z2
        with Znum.fast():
            r_an = z1 + z2
        lp_spread = max(r_lp.B) - min(r_lp.B)
        an_spread = max(r_an.B) - min(r_an.B)
        assert an_spread < lp_spread, (
            f"Analytical spread ({an_spread:.4f}) should be < "
            f"LP spread ({lp_spread:.4f})"
        )

    def test_paper_example4_full(self):
        """Full Znum pipeline matches Li et al. Example 4.

        Z1 = ((1,2,3), (0.7,0.8,0.9)), Z2 = ((7,8,9), (0.4,0.5,0.6))
        """
        z1 = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        z2 = Znum([7, 8, 8, 9], [0.4, 0.5, 0.5, 0.6])
        with Znum.fast():
            r = z1 + z2
        # A12 = (8, 10, 12)
        assert r.A[0] == pytest.approx(8.0)
        assert r.A[1] == pytest.approx(10.0)
        assert r.A[3] == pytest.approx(12.0)
        # B12: closed-form v12 = (v1+v2)/20 + (v1*v2)/15 + 3/5
        # Support: [417/619, 711/1000], peak: 83/120
        assert r.B[0] == pytest.approx(417 / 619, abs=0.005)
        assert r.B[1] == pytest.approx(83 / 120, abs=0.005)
        assert r.B[3] == pytest.approx(711 / 1000, abs=0.005)

    def test_low_b_inflation_is_inherent(self):
        """Low-B inflation is inherent: base value ≈ 0.6+ for symmetric triangles.

        This is NOT a bug — it's because Prob(X12 ∈ A12) is high when
        A12 support matches the convolution support. Documented behavior.
        """
        z1 = Znum([1, 2, 2, 3], [0.1, 0.2, 0.2, 0.3])
        z2 = Znum([1, 2, 2, 3], [0.1, 0.2, 0.2, 0.3])
        with Znum.fast():
            r = z1 + z2
        # B result will be ~0.61-0.64 (inherent to the method)
        assert all(0 <= b <= 1 for b in r.B)
        # Verify monotonicity
        assert r.B[0] <= r.B[1]
        assert r.B[2] <= r.B[3]


# ======================================================================
# Phase 6: Properties
# ======================================================================

class TestTriangularProperties:
    """Test is_triangular, A_tri, B_tri properties."""

    def test_triangular_detected(self):
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        assert z.is_triangular

    def test_non_triangular_A(self):
        z = Znum([1, 2, 3, 4], [0.7, 0.8, 0.8, 0.9])
        assert not z.is_triangular

    def test_non_triangular_B(self):
        z = Znum([1, 2, 2, 3], [0.1, 0.2, 0.3, 0.4])
        assert not z.is_triangular

    def test_A_tri_extraction(self):
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        a_tri = z.A_tri
        np.testing.assert_array_equal(a_tri, [1, 2, 3])

    def test_B_tri_extraction(self):
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        b_tri = z.B_tri
        np.testing.assert_array_equal(b_tri, [0.7, 0.8, 0.9])

    def test_A_tri_none_for_trapezoid(self):
        z = Znum([1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4])
        assert z.A_tri is None

    def test_B_tri_none_for_trapezoid(self):
        z = Znum([1, 2, 2, 3], [0.1, 0.2, 0.3, 0.4])
        assert z.B_tri is None

    def test_is_triangle_still_works(self):
        """Existing is_triangle property (A-only) still works."""
        z = Znum([1, 2, 2, 3], [0.1, 0.2, 0.3, 0.4])
        assert z.is_triangle  # A is triangular
        assert not z.is_triangular  # but B is not


# ======================================================================
# Phase 6: Context manager
# ======================================================================

class TestFastContextManager:
    """Test Znum.fast() context manager behavior."""

    def test_fast_resets_after_block(self):
        """Fast state resets after context exits."""
        from znum.math_ops import _state
        assert not getattr(_state, 'fast', False)
        with Znum.fast():
            assert getattr(_state, 'fast', False)
        assert not getattr(_state, 'fast', False)

    def test_fast_resets_on_exception(self):
        """Fast state resets even if exception occurs."""
        from znum.math_ops import _state
        try:
            with Znum.fast():
                raise RuntimeError("test error")
        except RuntimeError:
            pass
        assert not getattr(_state, 'fast', False)

    def test_nested_fast(self):
        """Nested fast blocks work correctly."""
        from znum.math_ops import _state
        with Znum.fast():
            assert getattr(_state, 'fast', False)
            with Znum.fast():
                assert getattr(_state, 'fast', False)
            assert getattr(_state, 'fast', False)
        assert not getattr(_state, 'fast', False)

    def test_radd_with_zero(self):
        """sum() uses __radd__ with 0 — must work in fast mode."""
        z = Znum([1, 2, 2, 3], [0.7, 0.8, 0.8, 0.9])
        with Znum.fast():
            result = sum([z, z])
        assert isinstance(result, Znum)
