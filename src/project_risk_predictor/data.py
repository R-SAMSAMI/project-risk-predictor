from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PROJECT_TYPES = [
    "Residential",
    "Commercial",
    "Industrial",
    "Infrastructure",
    "Renovation",
]
REGIONS = ["Northeast", "South", "Midwest", "West"]
CONTRACT_TYPES = ["Lump Sum", "Cost Plus", "GMP", "Unit Price"]
RISK_LEVELS = ["Low", "Medium", "High"]
COMPLEXITY_LEVELS = ["Low", "Medium", "High"]


@dataclass(frozen=True)
class ScenarioProfile:
    budget_scale: float
    duration_shift: int
    crew_shift: int
    base_delay_bias: float
    base_cost_bias: float


SCENARIO_PROFILES = {
    "Residential": ScenarioProfile(0.8, -30, 0, -0.25, -0.15),
    "Commercial": ScenarioProfile(1.0, 0, 2, 0.0, 0.05),
    "Industrial": ScenarioProfile(1.7, 45, 5, 0.35, 0.4),
    "Infrastructure": ScenarioProfile(2.0, 80, 8, 0.45, 0.25),
    "Renovation": ScenarioProfile(0.7, -20, -1, 0.15, 0.2),
}


def _sample_levels(rng: np.random.Generator, values: list[str], size: int, probs: list[float]) -> np.ndarray:
    return rng.choice(values, size=size, p=probs)


def generate_synthetic_projects(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    project_types = rng.choice(PROJECT_TYPES, size=n_samples, p=[0.23, 0.27, 0.16, 0.18, 0.16])
    regions = rng.choice(REGIONS, size=n_samples, p=[0.22, 0.31, 0.21, 0.26])
    contract_types = rng.choice(CONTRACT_TYPES, size=n_samples, p=[0.34, 0.17, 0.28, 0.21])

    rows: list[dict[str, object]] = []
    for idx in range(n_samples):
        project_type = project_types[idx]
        profile = SCENARIO_PROFILES[project_type]

        budget_musd = np.clip(
            rng.normal(24 * profile.budget_scale, 7 * profile.budget_scale),
            2,
            160,
        )
        planned_duration_days = int(
            np.clip(
                rng.normal(240 + profile.duration_shift, 75),
                60,
                950,
            )
        )
        crew_size = int(np.clip(rng.normal(32 + profile.crew_shift, 10), 6, 140))
        subcontractor_count = int(np.clip(rng.normal(7 + profile.crew_shift / 3, 3), 1, 25))
        change_order_count = int(np.clip(rng.poisson(2.5 + max(profile.base_delay_bias, 0) * 2), 0, 18))
        safety_incidents = int(np.clip(rng.poisson(0.6 + max(profile.base_delay_bias, 0.1)), 0, 8))
        permit_delay_days = int(np.clip(rng.normal(12 + profile.base_delay_bias * 8, 10), 0, 90))
        client_decision_latency = int(np.clip(rng.normal(6 + profile.base_delay_bias * 4, 3), 1, 25))
        weather_severity = _sample_levels(rng, RISK_LEVELS, 1, [0.35, 0.45, 0.20])[0]
        material_risk = _sample_levels(rng, RISK_LEVELS, 1, [0.28, 0.49, 0.23])[0]
        labor_availability = _sample_levels(rng, RISK_LEVELS, 1, [0.25, 0.52, 0.23])[0]
        site_complexity = _sample_levels(rng, COMPLEXITY_LEVELS, 1, [0.30, 0.48, 0.22])[0]
        site_density = _sample_levels(rng, COMPLEXITY_LEVELS, 1, [0.32, 0.44, 0.24])[0]
        percent_self_performed = float(np.clip(rng.normal(52, 18), 5, 95))
        equipment_utilization = float(np.clip(rng.normal(69, 12), 35, 98))

        weather_score = {"Low": 0.0, "Medium": 0.55, "High": 1.15}[weather_severity]
        material_score = {"Low": 0.0, "Medium": 0.70, "High": 1.35}[material_risk]
        labor_score = {"Low": -0.25, "Medium": 0.35, "High": 1.05}[labor_availability]
        complexity_score = {"Low": -0.15, "Medium": 0.45, "High": 1.15}[site_complexity]
        density_score = {"Low": -0.10, "Medium": 0.35, "High": 0.95}[site_density]
        contract_score = {
            "Lump Sum": 0.15,
            "Cost Plus": -0.10,
            "GMP": -0.05,
            "Unit Price": 0.18,
        }[contract_types[idx]]
        region_score = {
            "Northeast": 0.20,
            "South": 0.15,
            "Midwest": -0.05,
            "West": 0.12,
        }[regions[idx]]

        duration_pressure = max(planned_duration_days / max(crew_size, 1) - 7.5, 0)
        coordination_load = subcontractor_count / 8 + change_order_count / 4
        productivity_penalty = max(72 - equipment_utilization, 0) / 18
        self_perform_penalty = max(35 - percent_self_performed, 0) / 20

        latent_delay_score = (
            profile.base_delay_bias
            + weather_score
            + material_score
            + labor_score
            + complexity_score
            + density_score
            + contract_score
            + region_score
            + duration_pressure
            + coordination_load
            + permit_delay_days / 18
            + client_decision_latency / 9
            + safety_incidents * 0.45
            + productivity_penalty
            + self_perform_penalty
            + rng.normal(0, 0.65)
        )

        delay_probability = 1 / (1 + np.exp(-(latent_delay_score - 3.7)))
        delayed = int(rng.uniform() < delay_probability)
        delay_days = max(
            0,
            int(
                delayed
                * (
                    5
                    + latent_delay_score * 7.5
                    + change_order_count * 1.8
                    + permit_delay_days * 0.35
                    + rng.normal(0, 7)
                )
            ),
        )

        latent_cost_score = (
            profile.base_cost_bias
            + material_score * 1.2
            + complexity_score
            + density_score * 0.75
            + contract_score
            + coordination_load * 0.8
            + change_order_count * 0.22
            + delay_days / 35
            + safety_incidents * 0.35
            + budget_musd / 70
            + rng.normal(0, 0.5)
        )
        cost_overrun_pct = max(0.0, latent_cost_score * 2.3 + rng.normal(0, 2.4))
        over_budget = int(cost_overrun_pct >= 12.0)

        rows.append(
            {
                "project_type": project_type,
                "region": regions[idx],
                "contract_type": contract_types[idx],
                "budget_musd": round(float(budget_musd), 2),
                "planned_duration_days": planned_duration_days,
                "crew_size": crew_size,
                "subcontractor_count": subcontractor_count,
                "change_order_count": change_order_count,
                "safety_incidents": safety_incidents,
                "permit_delay_days": permit_delay_days,
                "client_decision_latency": client_decision_latency,
                "weather_severity": weather_severity,
                "material_risk": material_risk,
                "labor_availability": labor_availability,
                "site_complexity": site_complexity,
                "site_density": site_density,
                "percent_self_performed": round(percent_self_performed, 1),
                "equipment_utilization": round(equipment_utilization, 1),
                "delayed": delayed,
                "delay_days": delay_days,
                "cost_overrun_pct": round(cost_overrun_pct, 2),
                "over_budget": over_budget,
            }
        )

    return pd.DataFrame(rows)


def default_project_input() -> dict[str, object]:
    return {
        "project_type": "Commercial",
        "region": "South",
        "contract_type": "GMP",
        "budget_musd": 26.0,
        "planned_duration_days": 260,
        "crew_size": 34,
        "subcontractor_count": 8,
        "change_order_count": 2,
        "safety_incidents": 0,
        "permit_delay_days": 8,
        "client_decision_latency": 5,
        "weather_severity": "Medium",
        "material_risk": "Medium",
        "labor_availability": "Medium",
        "site_complexity": "Medium",
        "site_density": "Medium",
        "percent_self_performed": 54.0,
        "equipment_utilization": 72.0,
    }
