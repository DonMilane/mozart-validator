"""Shape Validator - a six-test gate for whether a backtested edge is real."""
from .validator import (RiskSpec, GateVerdict, validate_one_gate,
                        shape_validate, deflated_sharpe_pvalue)

__all__ = ["RiskSpec", "GateVerdict", "validate_one_gate", "shape_validate",
           "deflated_sharpe_pvalue"]
__version__ = "0.1.0"
