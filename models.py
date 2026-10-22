"""
Shared domain types used across the codebase.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class Coordinates:
    """Immutable geographic point with explicit lat/lon fields.

    Eliminates coordinate-swap bugs by replacing raw (lon, lat) / (lat, lon)
    tuples with named fields.  Every client accesses .lat and .lon directly
    and applies its own API convention internally.
    """
    lat: float
    lon: float

    @classmethod
    def from_dict(cls, d: dict) -> "Coordinates":
        """Create from a dict with latitude/longitude or lat/lon keys."""
        lat = d.get("latitude") or d.get("lat")
        lon = d.get("longitude") or d.get("lon")
        if lat is None or lon is None:
            raise ValueError(f"Cannot extract coordinates from {d}")
        return cls(lat=float(lat), lon=float(lon))


# ---------------------------------------------------------------------------
# Query Decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DecompositionStep:
    """One step in a decomposed compound query."""
    step_index: int
    sub_query: str
    required_capability: str
    depends_on: List[int] = field(default_factory=list)
    entity_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecomposedPlan:
    """Ordered plan produced by the router for compound queries."""
    original_query: str
    steps: List[DecompositionStep]



# ---------------------------------------------------------------------------
# Self-Reflection
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Rule-based quality assessment of specialist results."""
    score: float
    flags: List[str]
    covered_sub_intents: List[str] = field(default_factory=list)
    missing_sub_intents: List[str] = field(default_factory=list)

    @property
    def needs_reflection(self) -> bool:
        return self.score < 0.5


@dataclass
class ReflectionVerdict:
    """Output from LLM-based reflection on result quality."""
    action: str                                # PASS | REFORMULATE | PARTIAL
    reformulated_query: Optional[str] = None
    explanation: str = ""
    missing_info: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Environmental Reasoning Agent (ERA)
# ---------------------------------------------------------------------------

@dataclass
class ERASituation:
    """Step 0: situation archetype classification."""
    archetype: str                          # COMMUTE_PEAK | LEISURE | LATE_NIGHT | BAD_WEATHER | STANDARD
    archetype_reasoning: str                # why this archetype applies
    primary_dimensions: List[str]           # most influential for scoring
    secondary_dimensions: List[str]         # relevant but less decisive
    deprioritised_dimensions: List[str]     # unlikely to differentiate options


@dataclass
class ERADimensionAssessment:
    """ERA's assessment of a single environmental dimension."""
    salient: bool            # does this dimension meaningfully differentiate options?
    conditions: str          # what the current state is (or reason if not salient)
    impact: Optional[str]    # how it affects transport options (null if not salient)
    penalized_modes: List[str] = field(default_factory=list)
    severity: str = "none"   # none | low | moderate | high


@dataclass
class ERAOptionEval:
    """ERA's per-option evaluation with structured reasoning."""
    mode: str               # "transit" | "walking" | "driving" | "cycling"
    route_index: int        # 0..K-1 for transit, -1 walking, -2 driving, -3 cycling
    score: int              # 1-10 holistic score
    dimension_scores: Dict[str, Optional[int]]  # per-dimension scores (1-10, null if not salient)
    outdoor_minutes: float  # approximate outdoor exposure time
    weather_ok: bool        # weather does not significantly penalize this option
    safety_ok: bool         # no safety concerns (darkness, ice, etc.)
    reasoning: str          # synthesis of dimension scores into final judgment
    trade_off: Optional[str] = None  # what this option sacrifices vs best


@dataclass
class ERAOutput:
    """Output from the Environmental Reasoning Agent."""
    situation: ERASituation                              # step 0 classification
    dimension_analysis: Dict[str, ERADimensionAssessment]  # phase 1 reasoning
    options: List[ERAOptionEval]
    recommended_index: int           # route_index of recommended option
    recommendation_reason: str       # why this is the best choice
    condition_summary: str           # key environmental factors
    warnings: List[str] = field(default_factory=list)

    @property
    def best_route_index(self) -> int:
        return self.recommended_index

    @property
    def recommended_mode(self) -> str:
        for opt in self.options:
            if opt.route_index == self.recommended_index:
                return opt.mode
        return self.options[0].mode if self.options else "transit"
