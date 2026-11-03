
import math
from dataclasses import dataclass


@dataclass
class TransferQualityConfig:
    max_strength: float = 0.5
    error_low: float = 0.35
    error_high: float = 0.75
    uncertainty_penalty: float = 1.0
    margin: float = 0.0
    temperature: float = 0.1
    ema_decay: float = 0.9


class TransferQualityController:
    def __init__(self, config=None):
        self.config = config or TransferQualityConfig()
        self.smoothed_quality = None

    @staticmethod
    def clip(value, lower=0.0, upper=1.0):
        return max(lower, min(upper, value))

    def update(self, relative_error, quality, uncertainty):
        cfg = self.config

        values = (relative_error, quality, uncertainty)

        if not all(math.isfinite(float(v)) for v in values):
            return 0.0

        if cfg.error_high <= cfg.error_low:
            raise ValueError("error_high must exceed error_low")

        if cfg.temperature <= 0:
            raise ValueError("temperature must be positive")

        if uncertainty < 0:
            raise ValueError("uncertainty cannot be negative")

        reliability = self.clip(
            (cfg.error_high - relative_error)
            / (cfg.error_high - cfg.error_low)
        )

        if self.smoothed_quality is None:
            self.smoothed_quality = float(quality)
        else:
            self.smoothed_quality = (
                cfg.ema_decay * self.smoothed_quality
                + (1.0 - cfg.ema_decay) * float(quality)
            )

        adjusted_quality = (
            self.smoothed_quality
            - cfg.uncertainty_penalty * uncertainty
            - cfg.margin
        )

        z = adjusted_quality / cfg.temperature

        if z >= 0:
            sigmoid = 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            sigmoid = ez / (1.0 + ez)

        strength = cfg.max_strength * reliability * sigmoid

        return self.clip(strength, 0.0, cfg.max_strength)

    @staticmethod
    def effective_weights(raw_weights, strength):
        return 1.0 - strength * (1.0 - raw_weights)
