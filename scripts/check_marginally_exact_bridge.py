#!/usr/bin/env python3
"""Validate the marginally exact bridge coefficients.

This script checks that the implemented SI-space coefficients match the
coefficients obtained by:

1. matching SI time t in x_t = t x_1 + (1 - t) eps to the OU SNR,
2. writing the OU bridge over the remaining horizon,
3. rescaling back to SI coordinates.

It uses only the Python standard library so it can run in minimal
environments.
"""

from __future__ import annotations

import math


def implemented_coefficients(t_curr: float, t_next: float) -> tuple[float, float, float]:
    if not (0.0 <= t_curr <= t_next <= 1.0):
        raise ValueError("expected 0 <= t_curr <= t_next <= 1")
    if t_next == 0.0:
        raise ValueError("t_next = 0 is only valid for the trivial zero-step case")

    delta = t_next - t_curr
    remaining = 1.0 - t_curr
    mix_term = t_next * (1.0 - t_curr) + t_curr * (1.0 - t_next)

    coef_xt = t_curr * (1.0 - t_next) ** 2 / (t_next * remaining**2)
    coef_x1 = delta * mix_term / (t_next * remaining**2)
    var = (1.0 - t_next) ** 2 * delta * mix_term / (t_next**2 * remaining**2)
    return coef_xt, coef_x1, var


def d_term(t: float) -> float:
    return 1.0 - 2.0 * t + 2.0 * t * t


def si_scale(t: float) -> float:
    return math.sqrt(d_term(t))


def ou_horizon(t: float) -> float:
    """Matched OU time satisfying identical SNR."""
    return -math.log(t / math.sqrt(d_term(t)))


def transformed_ou_coefficients(t_curr: float, t_next: float) -> tuple[float, float, float]:
    h_curr = ou_horizon(t_curr)
    h_next = ou_horizon(t_next)
    a_curr = si_scale(t_curr)
    a_next = si_scale(t_next)

    sinh_h_curr = math.sinh(h_curr)
    sinh_h_next = math.sinh(h_next)
    sinh_gap = math.sinh(h_curr - h_next)

    coef_xt = a_next / a_curr * sinh_h_next / sinh_h_curr
    coef_x1 = a_next * sinh_gap / sinh_h_curr
    var = a_next * a_next * 2.0 * sinh_gap * sinh_h_next / sinh_h_curr
    return coef_xt, coef_x1, var


def assert_close(name: str, left: float, right: float, tol: float = 1e-10) -> None:
    if abs(left - right) > tol:
        raise AssertionError(f"{name}: {left} != {right} (tol={tol})")


def main() -> None:
    pairs = [
        (1e-4, 1e-3),
        (0.01, 0.10),
        (0.05, 0.20),
        (0.10, 0.50),
        (0.20, 0.80),
        (0.55, 0.90),
        (0.80, 0.95),
        (0.95, 0.999),
    ]

    for t_curr, t_next in pairs:
        impl = implemented_coefficients(t_curr, t_next)
        ref = transformed_ou_coefficients(t_curr, t_next)
        for label, left, right in zip(("coef_xt", "coef_x1", "var"), impl, ref):
            assert_close(f"{label}@({t_curr},{t_next})", left, right)

    boundary_pairs = [
        (0.0, 0.25),
        (0.10, 0.10),
        (0.40, 1.0),
    ]
    for t_curr, t_next in boundary_pairs:
        coef_xt, coef_x1, var = implemented_coefficients(t_curr, t_next)
        if t_curr == 0.0:
            assert_close("initial coef_xt", coef_xt, 0.0)
        if t_curr == t_next:
            assert_close("identity coef_xt", coef_xt, 1.0)
            assert_close("identity coef_x1", coef_x1, 0.0)
            assert_close("identity var", var, 0.0)
        if t_next == 1.0:
            assert_close("final coef_xt", coef_xt, 0.0)
            assert_close("final coef_x1", coef_x1, 1.0)
            assert_close("final var", var, 0.0)

    print("marginally exact bridge coefficients: OK")


if __name__ == "__main__":
    main()
