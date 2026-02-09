"""
Synthetic construction project dataset generator.

Generates 1000 realistic construction/engineering projects with embedded
non-linear relationships that make ML valuable over simple rules.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# Categorical feature options
PROJECT_TYPES = ["pharmaceutical", "biotech", "data_centre", "food_beverage", "medical_device"]
FACILITY_CLASSES = ["greenfield", "brownfield", "renovation"]
REGIONS = ["ireland", "uk", "mainland_europe", "north_america", "asia_pacific"]
COMPLEXITY_RATINGS = ["standard", "complex", "highly_complex"]
REGULATORY_ENVIRONMENTS = ["fda_regulated", "ema_regulated", "both", "non_regulated"]
PROCUREMENT_ROUTES = ["epcm", "design_build", "traditional", "construction_management"]
SITE_CONDITIONS = ["flat_urban", "flat_rural", "sloped", "constrained_industrial"]


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic construction project dataset.

    Args:
        n_samples: Number of projects to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with project features and duration_weeks target.
    """
    rng = np.random.default_rng(seed)

    # --- Categorical features ---
    project_type = rng.choice(PROJECT_TYPES, n_samples)
    facility_class = rng.choice(FACILITY_CLASSES, n_samples, p=[0.4, 0.35, 0.25])
    region = rng.choice(REGIONS, n_samples)
    complexity_rating = rng.choice(COMPLEXITY_RATINGS, n_samples, p=[0.35, 0.40, 0.25])
    regulatory_environment = rng.choice(REGULATORY_ENVIRONMENTS, n_samples, p=[0.30, 0.25, 0.20, 0.25])
    procurement_route = rng.choice(PROCUREMENT_ROUTES, n_samples)
    site_condition = rng.choice(SITE_CONDITIONS, n_samples)

    # --- Numerical features ---
    budget_millions = rng.uniform(5, 500, n_samples)
    building_area_sqm = rng.uniform(500, 50_000, n_samples)
    num_floors = rng.integers(1, 9, n_samples)  # 1-8
    cleanroom_percentage = np.zeros(n_samples)
    for i in range(n_samples):
        if project_type[i] in ("pharmaceutical", "biotech", "medical_device"):
            cleanroom_percentage[i] = rng.uniform(10, 80)
        elif project_type[i] == "food_beverage":
            cleanroom_percentage[i] = rng.uniform(0, 30)
        else:
            cleanroom_percentage[i] = rng.uniform(0, 10)

    num_stakeholders = rng.integers(3, 26, n_samples)
    team_size = rng.integers(20, 501, n_samples)
    design_completion_pct = rng.uniform(30, 100, n_samples)
    num_change_orders = rng.integers(0, 41, n_samples)

    # --- Binary features ---
    includes_cqv = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if project_type[i] in ("pharmaceutical", "biotech", "medical_device"):
            includes_cqv[i] = rng.choice([0, 1], p=[0.15, 0.85])
        else:
            includes_cqv[i] = rng.choice([0, 1], p=[0.75, 0.25])

    has_bim = rng.choice([0, 1], n_samples, p=[0.3, 0.7])
    is_modular = rng.choice([0, 1], n_samples, p=[0.7, 0.3])

    # --- Target: duration_weeks ---
    # Build up from components with realistic relationships.
    # Key: non-linear interactions that tree models can capture but linear can't.

    # Base duration from area (square root = diminishing returns)
    base_duration = 15 + np.sqrt(building_area_sqm) * 0.5 + num_floors * 4

    # Budget contribution (log = diminishing returns)
    base_duration += np.log1p(budget_millions) * 6

    # Project type multipliers (applied multiplicatively to base)
    type_multipliers = {
        "pharmaceutical": 1.45,
        "biotech": 1.35,
        "data_centre": 0.80,
        "food_beverage": 1.00,
        "medical_device": 1.15,
    }
    type_mult = np.array([type_multipliers[t] for t in project_type])
    duration = base_duration * type_mult

    # Complexity interacts with project type (complex pharma is much worse than complex data centre)
    complexity_mult = {"standard": 1.0, "complex": 1.12, "highly_complex": 1.28}
    duration *= np.array([complexity_mult[c] for c in complexity_rating])

    # Facility class
    facility_add = {"greenfield": 0, "brownfield": 10, "renovation": 20}
    duration += np.array([facility_add[f] for f in facility_class])

    # Regulatory environment - interacts with cleanroom percentage
    reg_base = {"non_regulated": 0, "fda_regulated": 12, "ema_regulated": 10, "both": 22}
    reg_effect = np.array([reg_base[r] for r in regulatory_environment])
    # Regulation hurts more when there's more cleanroom area
    reg_cleanroom_interaction = reg_effect * (1 + cleanroom_percentage / 100)
    duration += reg_cleanroom_interaction

    # Cleanroom * CQV interaction (multiplicative, not just additive)
    # This is a key non-linear interaction: cleanroom area + CQV validation
    cqv_effect = (cleanroom_percentage / 100) * building_area_sqm * includes_cqv * 0.002
    duration += cqv_effect

    # Design completion - non-linear inverse effect
    # Going from 30% to 50% saves much more time than 80% to 100%
    design_penalty = 50 * np.exp(-0.04 * design_completion_pct)
    duration += design_penalty

    # Change orders interact with incomplete design (strong interaction)
    change_impact = num_change_orders * (1 - design_completion_pct / 100) ** 1.5 * 4
    duration += change_impact

    # Modular construction: only helps on greenfield, and benefit scales with area
    is_greenfield = (facility_class == "greenfield").astype(int)
    modular_benefit = is_modular * is_greenfield * np.sqrt(building_area_sqm) * (-0.1)
    duration += modular_benefit

    # BIM saves time proportional to complexity
    bim_savings = {"standard": 3, "complex": 8, "highly_complex": 14}
    duration -= has_bim * np.array([bim_savings[c] for c in complexity_rating])

    # Stakeholder overhead (quadratic - each extra stakeholder adds more overhead)
    duration += num_stakeholders * 0.5 + (num_stakeholders ** 1.5) * 0.15

    # Team size: larger teams reduce duration but with strongly diminishing returns
    duration -= np.log1p(team_size) * 3

    # Procurement route
    proc_add = {"epcm": 0, "design_build": -8, "traditional": 10, "construction_management": 4}
    duration += np.array([proc_add[p] for p in procurement_route])

    # Site condition interacts with building area (constrained sites hurt more for big builds)
    site_mult = {"flat_urban": 1.0, "flat_rural": 0.98, "sloped": 1.04, "constrained_industrial": 1.08}
    site_factor = np.array([site_mult[s] for s in site_condition])
    duration *= site_factor

    # === Higher-order interactions that feature engineering doesn't capture ===
    # These threshold/conditional effects give tree models an advantage.

    # Three-way: pharma + high cleanroom + FDA/both regulation = massive CQV overhead
    is_pharma_or_bio = np.isin(project_type, ["pharmaceutical", "biotech"]).astype(float)
    is_strict_reg = np.isin(regulatory_environment, ["fda_regulated", "both"]).astype(float)
    pharma_reg_cleanroom = is_pharma_or_bio * is_strict_reg * (cleanroom_percentage / 100) * 35
    duration += pharma_reg_cleanroom

    # Threshold: large buildings (>20000 sqm) have coordination overhead that jumps
    large_building_penalty = np.where(building_area_sqm > 20000, 25 + (building_area_sqm - 20000) * 0.001, 0)
    duration += large_building_penalty

    # Threshold: high change orders (>20) on complex projects cause cascading delays
    is_complex = np.isin(complexity_rating, ["complex", "highly_complex"]).astype(float)
    high_change_cascade = np.where(num_change_orders > 20, (num_change_orders - 20) * is_complex * 3, 0)
    duration += high_change_cascade

    # Budget-area ratio: premium builds (high spend per sqm) take longer, threshold effect
    budget_per_sqm = budget_millions * 1e6 / building_area_sqm
    premium_build = np.where(budget_per_sqm > 20000, np.sqrt(np.maximum(budget_per_sqm - 20000, 0)) * 0.3, 0)
    duration += premium_build

    # Team size has a sweet spot: very small teams (<50) are slow, very large (>300) have overhead
    small_team_penalty = np.where(team_size < 50, (50 - team_size) * 0.8, 0)
    large_team_overhead = np.where(team_size > 300, (team_size - 300) * 0.15, 0)
    duration += small_team_penalty + large_team_overhead

    # Design-build procurement gets extra benefit on standard complexity
    is_design_build = (procurement_route == "design_build").astype(float)
    is_standard = (complexity_rating == "standard").astype(float)
    db_standard_bonus = is_design_build * is_standard * (-15)
    duration += db_standard_bonus

    # Renovation + constrained site is particularly painful
    is_renovation = (facility_class == "renovation").astype(float)
    is_constrained = (site_condition == "constrained_industrial").astype(float)
    renovation_constrained = is_renovation * is_constrained * 25
    duration += renovation_constrained

    # Asia Pacific pharma projects face additional supply chain delays
    is_apac = (region == "asia_pacific").astype(float)
    apac_pharma = is_apac * is_pharma_or_bio * 15
    duration += apac_pharma

    # Add Gaussian noise (~7% of duration)
    noise = rng.normal(0, duration * 0.07)
    duration += noise

    # Floor duration at minimum 8 weeks
    duration = np.maximum(duration, 8)

    # Round to 1 decimal
    duration = np.round(duration, 1)

    # --- Assemble DataFrame ---
    df = pd.DataFrame({
        "project_type": project_type,
        "facility_class": facility_class,
        "region": region,
        "complexity_rating": complexity_rating,
        "regulatory_environment": regulatory_environment,
        "procurement_route": procurement_route,
        "site_condition": site_condition,
        "budget_millions": np.round(budget_millions, 2),
        "building_area_sqm": np.round(building_area_sqm, 1),
        "num_floors": num_floors,
        "cleanroom_percentage": np.round(cleanroom_percentage, 1),
        "num_stakeholders": num_stakeholders,
        "team_size": team_size,
        "design_completion_pct": np.round(design_completion_pct, 1),
        "num_change_orders": num_change_orders,
        "includes_cqv": includes_cqv,
        "has_bim": has_bim,
        "is_modular": is_modular,
        "duration_weeks": duration,
    })

    return df


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    df = generate_dataset()
    output_path = data_dir / "construction_projects.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} projects -> {output_path}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"\nTarget stats:")
    print(f"  Mean duration: {df['duration_weeks'].mean():.1f} weeks")
    print(f"  Std deviation: {df['duration_weeks'].std():.1f} weeks")
    print(f"  Min: {df['duration_weeks'].min():.1f}, Max: {df['duration_weeks'].max():.1f}")
    print(f"\nNull values: {df.isnull().sum().sum()}")
