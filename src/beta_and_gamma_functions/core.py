"""Log-gamma and beta functions via the Lanczos approximation.

The Lanczos approximation is a well-known method for computing the gamma
function with high accuracy for Re(z) > 0. We extend it to the whole complex
plane using the reflection formula

    gamma(z) * gamma(1 - z) = pi / sin(pi * z)

so that

    log|gamma(z)| = log(pi) - log|sin(pi * z)| - Re(loggamma(1 - z))

for Re(z) <= 0.5. The branch cut of cmath.log means we return the principal
value of loggamma; the imaginary part can jump across the negative real axis.
This is the standard convention used by SciPy's ``loggamma`` for complex
input.

The real-valued ``lgamma`` returns only the real part (the log-modulus), which
is single-valued and continuous away from the non-positive integers where
gamma has simple poles.
"""

import cmath
import math

# Lanczos coefficients for g = 7, n = 9, as popularized by Numerical Recipes.
# These give roughly 15 digits of accuracy for Re(z) > 0.5.
_LANCZOS_G = 7.0
_LANCZOS_C = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)


def _lanczos_sum(z):
    """Evaluate the Lanczos series sum for z, returning a complex number.

    The sum is c[0] + sum_{k=1}^{n-1} c[k] / (z + k). We compute it with a
    straightforward loop; the number of terms is tiny (9) so Horner-style
    rearrangement buys nothing meaningful and hurts readability.
    """
    acc = complex(_LANCZOS_C[0])
    for k in range(len(_LANCZOS_C) - 1, 0, -1):
        acc += _LANCZOS_C[k] / (z + k)
    return acc


def _loggamma_lanczos(z):
    """Principal loggamma for Re(z) > 0.5 via the Lanczos approximation.

    Returns a complex number. The caller is responsible for ensuring the
    input lies in the right half-plane; no check is performed here so that
    the reflection logic in ``_loggamma_complex`` stays in one place.
    """
    z = complex(z)
    # Shift so the argument to the series is z - 1, as the standard form
    # requires. The (z + g - 0.5) factor outside comes from Stirling's
    # asymptotic shape that Lanczos bakes in.
    t = z + _LANCZOS_G - 0.5
    a = _lanczos_sum(z - 1.0)
    # cmath.log gives the principal branch; this is where the branch cut of
    # the complex loggamma is introduced.
    return (z - 0.5) * cmath.log(t) - t + cmath.log(2.0 * math.pi) / 2.0 + cmath.log(a)


def _loggamma_complex(z):
    """Principal loggamma for any complex z not a non-positive integer.

    Uses reflection for Re(z) <= 0.5. Non-positive integers are poles of
    gamma, so loggamma is undefined there; we raise ValueError.
    """
    z = complex(z)

    # Detect non-positive integers: poles of gamma. We check both real and
    # imaginary parts with a tolerance, because a user may pass -2.0 exactly.
    if abs(z.imag) < 1e-14 and abs(z.real - round(z.real)) < 1e-14 and z.real <= 0.0:
        raise ValueError(
            "loggamma is undefined at non-positive integers; z = %r" % (z,)
        )

    if z.real > 0.5:
        return _loggamma_lanczos(z)

    # Reflection: gamma(z) = pi / (sin(pi*z) * gamma(1 - z))
    # => loggamma(z) = log(pi) - log(sin(pi*z)) - loggamma(1 - z)
    # The branch of the logarithm here is inherited from cmath.log, which is
    # the principal branch. This is the source of the imaginary-part
    # discontinuity across the negative real axis.
    one_minus_z = 1.0 - z
    sin_pi_z = cmath.sin(math.pi * z)
    return cmath.log(math.pi) - cmath.log(sin_pi_z) - _loggamma_lanczos(one_minus_z)


def lgamma(z):
    """Natural log of |gamma(z)|, the standard real-valued lgamma.

    Accepts a real number (int or float) or a complex number. For real
    input the return type is float. For complex input the return type is
    also float, because we return the log-modulus — the real part of the
    principal loggamma. This matches the behaviour of math.lgamma for real
    arguments and extends it sensibly to the complex plane.

    Raises ValueError at non-positive integers, where gamma has poles.
    """
    if isinstance(z, complex):
        return _loggamma_complex(z).real
    # Treat bool as int (Python does this anyway) but be explicit: bool is a
    # subclass of int, and lgamma(True) should equal lgamma(1) = 0.0.
    if isinstance(z, (int, float)):
        # math.lgamma is faster, standard, and handles the real case including
        # the sign-flip behaviour at non-positive integers by raising ValueError
        # for non-positive integers and returning a float otherwise. We delegate
        # to it rather than re-implementing real-path reflection.
        return float(math.lgamma(z))
    # Allow anything that can be coerced to complex, e.g. fractions.Fraction.
    try:
        cz = complex(z)
    except (TypeError, ValueError):
        raise TypeError(
            "lgamma expected a real or complex number, got %s" % type(z).__name__
        )
    return _loggamma_complex(cz).real


def log_beta(a, b):
    """Natural log of |beta(a, b)| = |gamma(a) gamma(b) / gamma(a + b)|.

    Returns a float: the log-modulus of the beta function. This is the
    numerically stable way to compute beta for large arguments, and it is
    the quantity most statistical applications actually need.

    Raises ValueError if a, b, or a+b is a non-positive integer.
    """
    # We go through the complex loggamma so that complex inputs are supported
    # and so that the reflection logic is handled uniformly. For real inputs
    # this is a minor overhead but keeps one code path.
    la = _loggamma_complex(complex(a))
    lb = _loggamma_complex(complex(b))
    lab = _loggamma_complex(complex(a) + complex(b))
    return (la + lb - lab).real


def beta(a, b):
    """The beta function B(a, b) = gamma(a) gamma(b) / gamma(a + b).

    Returns a float for real inputs and a complex number for complex inputs.
    For large arguments the result may underflow to 0.0; prefer log_beta in
    that regime.

    Raises ValueError if a, b, or a+b is a non-positive integer.
    """
    a_c = complex(a)
    b_c = complex(b)
    la = _loggamma_complex(a_c)
    lb = _loggamma_complex(b_c)
    lab = _loggamma_complex(a_c + b_c)
    log_b = la + lb - lab
    result = cmath.exp(log_b)
    # If both inputs were real, return a float so callers don't have to peel
    # off a zero imaginary part. We check the *original* arguments, not the
    # coerced complex forms, so that passing 2+3j deliberately yields complex.
    if not isinstance(a, complex) and not isinstance(b, complex):
        return result.real
    return result
