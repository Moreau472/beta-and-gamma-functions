import math
import unittest

from beta_and_gamma_functions import lgamma, beta, log_beta


class TestLgammaReal(unittest.TestCase):
    """lgamma on real inputs should agree with math.lgamma."""

    def test_lgamma_one(self):
        self.assertAlmostEqual(lgamma(1.0), 0.0, places=12)

    def test_lgamma_two(self):
        # gamma(2) = 1! = 1, so log is 0.
        self.assertAlmostEqual(lgamma(2.0), 0.0, places=12)

    def test_lgamma_half(self):
        # gamma(1/2) = sqrt(pi), so lgamma = 0.5 * log(pi).
        expected = 0.5 * math.log(math.pi)
        self.assertAlmostEqual(lgamma(0.5), expected, places=12)

    def test_lgamma_ten(self):
        # gamma(10) = 9! = 362880.
        expected = math.log(362880.0)
        self.assertAlmostEqual(lgamma(10.0), expected, places=10)

    def test_lgamma_large(self):
        # Spot-check a larger value against math.lgamma.
        self.assertAlmostEqual(lgamma(50.5), math.lgamma(50.5), places=8)

    def test_lgamma_negative_non_integer(self):
        # gamma(-0.5) = -2*sqrt(pi), so |gamma(-0.5)| = 2*sqrt(pi).
        expected = math.log(2.0) + 0.5 * math.log(math.pi)
        self.assertAlmostEqual(lgamma(-0.5), expected, places=12)

    def test_lgamma_int_input(self):
        # int should be accepted and match the float path.
        self.assertAlmostEqual(lgamma(5), math.lgamma(5), places=12)

    def test_lgamma_non_positive_integer_raises(self):
        for n in (0, -1, -2, -3):
            with self.assertRaises(ValueError):
                lgamma(n)

    def test_lgamma_bool_input(self):
        # bool is a subclass of int; lgamma(True) == lgamma(1) == 0.0.
        self.assertAlmostEqual(lgamma(True), 0.0, places=12)

    def test_lgamma_type_error_on_string(self):
        with self.assertRaises(TypeError):
            lgamma("hello")


class TestLgammaComplex(unittest.TestCase):
    """lgamma on complex inputs returns the log-modulus (a float)."""

    def test_lgamma_complex_i(self):
        # Compare against a known high-precision value.
        # loggamma(1j) ≈ -0.6509233503... + 4.747...j  (principal value)
        # |gamma(1j)| = exp(Re loggamma(1j))
        expected_real = -0.6509233503144414
        self.assertAlmostEqual(lgamma(1j), expected_real, places=6)

    def test_lgamma_complex_pure_imaginary_consistency(self):
        # gamma(conj(z)) = conj(gamma(z)) for the principal branch when z is
        # not on the branch cut, so |gamma(z)| = |gamma(conj(z))|.
        z = 2.0 + 3.0j
        self.assertAlmostEqual(lgamma(z), lgamma(z.conjugate()), places=12)

    def test_lgamma_complex_returns_float(self):
        self.assertIsInstance(lgamma(1j), float)

    def test_lgamma_complex_at_pole_raises(self):
        with self.assertRaises(ValueError):
            lgamma(-2 + 0j)


class TestBeta(unittest.TestCase):
    """beta and log_beta against known identities."""

    def test_beta_symmetry(self):
        # B(a, b) = B(b, a).
        self.assertAlmostEqual(beta(2.0, 3.0), beta(3.0, 2.0), places=12)

    def test_beta_integer_identity(self):
        # B(m, n) = (m-1)!(n-1)! / (m+n-1)! for positive integers.
        # B(3, 4) = 2! * 3! / 6! = 2 * 6 / 720 = 12/720 = 1/60.
        self.assertAlmostEqual(beta(3, 4), 1.0 / 60.0, places=12)

    def test_beta_half_identity(self):
        # B(1/2, 1/2) = pi.
        self.assertAlmostEqual(beta(0.5, 0.5), math.pi, places=10)

    def test_beta_returns_float_for_real_inputs(self):
        self.assertIsInstance(beta(2.0, 3.0), float)

    def test_beta_returns_complex_for_complex_input(self):
        result = beta(1j, 2.0)
        self.assertIsInstance(result, complex)

    def test_log_beta_matches_log_of_beta(self):
        # For real positive arguments, log_beta(a, b) == log(beta(a, b)).
        a, b = 2.5, 3.5
        self.assertAlmostEqual(log_beta(a, b), math.log(beta(a, b)), places=10)

    def test_log_beta_large_arguments(self):
        # beta(100, 100) is very small but does not underflow to 0.0 in
        # double precision; log_beta should be a sane negative number.
        lb = log_beta(100.0, 100.0)
        self.assertLess(lb, -100.0)
        # beta(100, 100) ≈ 2.2e-61, which is representable as a float.
        self.assertGreater(beta(100.0, 100.0), 0.0)

    def test_beta_at_pole_raises(self):
        # a = 0 is a pole of gamma.
        with self.assertRaises(ValueError):
            beta(0.0, 2.0)

    def test_beta_sum_pole_raises(self):
        # a + b = -1, a pole of gamma(a+b).
        with self.assertRaises(ValueError):
            beta(1.0, -2.0)

    def test_log_beta_returns_float(self):
        self.assertIsInstance(log_beta(2.0, 3.0), float)


if __name__ == "__main__":
    unittest.main()
