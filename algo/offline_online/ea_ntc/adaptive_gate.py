"""Training-only adaptive source-gate strength controller."""

import math


class AdaptiveGateController:
    def __init__(
        self,
        max_strength=0.50,
        error_low=0.35,
        error_high=0.75,
    ):
        self.max_strength = float(max_strength)
        self.error_low = float(error_low)
        self.error_high = float(error_high)

        values = (
            self.max_strength,
            self.error_low,
            self.error_high,
        )

        if not all(math.isfinite(x) for x in values):
            raise ValueError("Controller parameters must be finite")

        if not 0.0 <= self.max_strength <= 1.0:
            raise ValueError("max_strength must be in [0, 1]")

        if not 0.0 <= self.error_low < self.error_high:
            raise ValueError(
                "Require 0 <= error_low < error_high"
            )

    def strength(self, relative_error, ready):
        """Return a strength using training-only diagnostics.

        An unavailable or unreliable model disables filtering.
        """

        if not ready or relative_error is None:
            return 0.0

        error = float(relative_error)

        if not math.isfinite(error) or error < 0.0:
            return 0.0

        fraction = (
            (self.error_high - error)
            / (self.error_high - self.error_low)
        )

        fraction = max(0.0, min(1.0, fraction))

        return self.max_strength * fraction
