"""
Environmental Reasoning Agent (ERA) for evaluating route options against
real-time environmental conditions (weather, traffic, time of day, parking).
Produces structured per-option reasoning that the Synthesizer presents to users.
"""

from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentValidationError
from models import ERAOutput, ERAOptionEval, ERADimensionAssessment, ERASituation
from prompts.era_prompts import build_era_prompt, validate_era_output


class ERAAgent(BaseAgent[ERAOutput]):

    def __init__(
        self,
        client: Any,
        model: str = "qwen3-30b-a3b-instruct-2507",
        timeout: int = 8,
        max_retries: int = 1,
        retry_delay: float = 0.5,
        temperature: float = 0.1,
        max_tokens: int = 2500,
        verbose: bool = False
    ):
        super().__init__(
            name="ERAAgent",
            model=model,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            temperature=temperature,
            max_tokens=max_tokens,
            verbose=verbose
        )

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        if not isinstance(input_data, dict):
            raise AgentValidationError("Input must be a dictionary")

        if "candidates" not in input_data:
            raise AgentValidationError("Input must contain 'candidates' key")

        candidates = input_data["candidates"]
        if not isinstance(candidates, list):
            raise AgentValidationError("'candidates' must be a list")

        has_alternatives = any(
            input_data.get(k) for k in ("walking_alternative", "driving_alternative", "cycling_alternative")
        )
        if len(candidates) == 0 and not has_alternatives:
            raise AgentValidationError("No options to evaluate: candidates list is empty and no alternatives provided")

    def _execute_internal(self, input_data: Dict[str, Any]) -> str:
        candidates = input_data["candidates"]
        walking_alt = input_data.get("walking_alternative")
        driving_alt = input_data.get("driving_alternative")
        cycling_alt = input_data.get("cycling_alternative")
        proactive_context = input_data.get("proactive_context")

        n_options = len(candidates)
        if walking_alt:
            n_options += 1
        if driving_alt:
            n_options += 1
        if cycling_alt:
            n_options += 1
        self._log(f"Evaluating {n_options} options against environmental conditions")

        messages = build_era_prompt(
            candidates=candidates,
            walking_alternative=walking_alt,
            driving_alternative=driving_alt,
            cycling_alternative=cycling_alt,
            proactive_context=proactive_context,
        )

        response = self._call_llm(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return response

    def _format_output(self, raw_output: str) -> ERAOutput:
        parsed = self._parse_json_response(raw_output)

        is_valid, error_msg = validate_era_output(parsed)
        if not is_valid:
            raise AgentValidationError(f"ERA output validation failed: {error_msg}")

        # Parse Step 0: situation classification
        sit = parsed["situation"]
        situation = ERASituation(
            archetype=str(sit["archetype"]),
            archetype_reasoning=str(sit["archetype_reasoning"]),
            primary_dimensions=sit.get("primary_dimensions", []),
            secondary_dimensions=sit.get("secondary_dimensions", []),
            deprioritised_dimensions=sit.get("deprioritised_dimensions", []),
        )

        # Parse Phase 1: dimension analysis
        dimension_analysis = {}
        for dim_name, dim_data in parsed["dimension_analysis"].items():
            dimension_analysis[dim_name] = ERADimensionAssessment(
                salient=bool(dim_data.get("salient", True)),
                conditions=str(dim_data.get("conditions", "")),
                impact=dim_data.get("impact"),
                penalized_modes=dim_data.get("penalized_modes", []),
                severity=str(dim_data.get("severity", "none")),
            )

        # Parse Phase 2: per-option evaluations
        options = []
        for opt in parsed["options"]:
            raw_dim_scores = opt.get("dimension_scores", {})
            dimension_scores = {
                k: int(v) if v is not None else None
                for k, v in raw_dim_scores.items()
            }

            options.append(ERAOptionEval(
                mode=opt["mode"],
                route_index=int(opt["route_index"]),
                score=int(opt["score"]),
                dimension_scores=dimension_scores,
                outdoor_minutes=float(opt.get("outdoor_minutes", 0)),
                weather_ok=bool(opt.get("weather_ok", True)),
                safety_ok=bool(opt.get("safety_ok", True)),
                reasoning=str(opt["reasoning"]),
                trade_off=opt.get("trade_off"),
            ))

        return ERAOutput(
            situation=situation,
            dimension_analysis=dimension_analysis,
            options=options,
            recommended_index=int(parsed["recommended_index"]),
            recommendation_reason=str(parsed["recommendation_reason"]),
            condition_summary=str(parsed["condition_summary"]),
            warnings=parsed.get("warnings", []),
        )

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        walking_alt: Optional[Dict[str, Any]] = None,
        driving_alt: Optional[Dict[str, Any]] = None,
        cycling_alt: Optional[Dict[str, Any]] = None,
        proactive_context: Optional[Dict[str, Any]] = None,
    ) -> ERAOutput:
        """Convenience method for evaluating route options."""
        return self.execute({
            "candidates": candidates,
            "walking_alternative": walking_alt,
            "driving_alternative": driving_alt,
            "cycling_alternative": cycling_alt,
            "proactive_context": proactive_context,
        })


def create_era_agent(
    client: Any,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> ERAAgent:
    if config is None:
        config = {}

    return ERAAgent(
        client=client,
        model=config.get("model", "qwen3-30b-a3b-instruct-2507"),
        timeout=config.get("timeout", 8),
        max_retries=config.get("max_retries", 1),
        retry_delay=config.get("retry_delay", 0.5),
        temperature=config.get("temperature", 0.1),
        max_tokens=config.get("max_tokens", 2500),
        verbose=verbose,
    )
