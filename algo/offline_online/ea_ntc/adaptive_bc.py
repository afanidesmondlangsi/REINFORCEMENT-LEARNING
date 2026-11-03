import math


class AdaptiveBCController:
    """Conservative, uncertainty-aware source BC coefficient."""

    def __init__(
        self,
        confidence_multiplier=2.0,
        margin=0.0,
        uncertainty_scale=2.0,
        max_coefficient=1.0,
        uncertainty_floor=1e-12,
    ):
        self.confidence_multiplier = float(confidence_multiplier)
        self.margin = float(margin)
        self.uncertainty_scale = float(uncertainty_scale)
        self.max_coefficient = float(max_coefficient)
        self.uncertainty_floor = float(uncertainty_floor)

        values = (
            self.confidence_multiplier,
            self.margin,
            self.uncertainty_scale,
            self.max_coefficient,
            self.uncertainty_floor,
        )

        if not all(math.isfinite(x) for x in values):
            raise ValueError("BC controller parameters must be finite")

        if (
            self.confidence_multiplier < 0
            or self.margin < 0
            or self.uncertainty_scale <= 0
            or not 0 <= self.max_coefficient <= 1
            or self.uncertainty_floor <= 0
        ):
            raise ValueError("Invalid adaptive BC configuration")

        self.last_lower_bound = None
        self.last_coefficient = 0.0

    def update(self, quality, uncertainty):
        if quality is None or uncertainty is None:
            self.last_lower_bound = None
            self.last_coefficient = 0.0
            return 0.0

        quality = float(quality)
        uncertainty = float(uncertainty)

        if not math.isfinite(quality) or not math.isfinite(uncertainty):
            self.last_lower_bound = None
            self.last_coefficient = 0.0
            return 0.0

        if uncertainty < 0:
            self.last_lower_bound = None
            self.last_coefficient = 0.0
            return 0.0

        lower_bound = (
            quality
            - self.confidence_multiplier * uncertainty
            - self.margin
        )
        self.last_lower_bound = lower_bound

        if lower_bound <= 0:
            coefficient = 0.0
        else:
            denominator = (
                lower_bound
                + self.uncertainty_scale
                * max(uncertainty, self.uncertainty_floor)
            )
            coefficient = (
                self.max_coefficient * lower_bound / denominator
            )

        self.last_coefficient = coefficient
        return coefficient
