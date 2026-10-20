# beta-and-gamma-functions

A small Python library computing the log-gamma and beta functions for real and complex arguments, using the Lanczos approximation and the standard-library `cmath` module. No third-party dependencies.

## Usage

```python
from beta_and_gamma_functions import lgamma, beta, log_beta

# Real log-gamma; matches math.lgamma but also accepts complex input.
print(lgamma(0.5))        # 0.5723649429247001

# Complex log-modulus of gamma.
print(lgamma(1j))         # -0.6509233503...

# The beta function for real arguments returns a float.
print(beta(0.5, 0.5))     # 3.141592653589793 (i.e. pi)

# log|beta| is stable for large arguments where beta underflows to 0.
print(log_beta(100.0, 100.0))
```

## Why this exists

`math.lgamma` handles real arguments but not complex ones. SciPy's `loggamma` handles both but pulls in a large dependency. This library fills the gap for code that needs log-gamma or beta on complex inputs but cannot or will not depend on NumPy/SciPy — for example, inside a constrained container or a pure-stdlib service.

The trade-off is precision and branch-cut sophistication: the Lanczos approximation used here (g=7, n=9) gives roughly 15 digits of accuracy for `Re(z) > 0.5`, and reflection extends it to the rest of the complex plane. That is enough for almost all statistical and numerical work, but it is not arbitrary-precision.

## Exported names

- `lgamma(z)` — natural log of `|gamma(z)|`. Accepts `int`, `float`, or `complex`. Returns `float`. Raises `ValueError` at non-positive integers.
- `beta(a, b)` — the beta function `gamma(a)gamma(b)/gamma(a+b)`. Returns `float` when both inputs are real, `complex` otherwise. Raises `ValueError` if `a`, `b`, or `a+b` is a non-positive integer.
- `log_beta(a, b)` — natural log of `|beta(a, b)|`. Returns `float`. Raises `ValueError` under the same conditions as `beta`.

## Awkward edge

The complex log-gamma is multi-valued. This library returns the **principal value**, which means the imaginary part of the internal computation can jump discontinuously across the negative real axis. `lgamma` sidesteps this by returning only the real part (the log-modulus), which is single-valued away from the poles. If you need the full complex `loggamma` with a specific branch, this library does not expose that — use SciPy.

At non-positive integers (`0, -1, -2, ...`) the gamma function has simple poles, and all three functions raise `ValueError`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

