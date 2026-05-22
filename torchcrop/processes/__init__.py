"""Biophysical process modules."""

from torchcrop.processes.astro import Astro
from torchcrop.processes.co2_transpiration import Co2Transpiration
from torchcrop.processes.evapotranspiration import PotentialEvapoTranspiration
from torchcrop.processes.heat_stress import (
    HeatStressOnGrain,
    HeatStressOnLeafSenescence,
)
from torchcrop.processes.irradiation import Irradiation
from torchcrop.processes.leaf_dynamics import LeafDynamics
from torchcrop.processes.nutrient_demand import NutrientDemand
from torchcrop.processes.partitioning import Partitioning
from torchcrop.processes.phenology import Phenology
from torchcrop.processes.photosynthesis import Photosynthesis
from torchcrop.processes.root_dynamics import RootDynamics
from torchcrop.processes.soil_nutrients import SoilNutrients
from torchcrop.processes.stem_dynamics import StemDynamics
from torchcrop.processes.stress import StressFactors
from torchcrop.processes.water_balance import WaterBalance

__all__ = [
    "Astro",
    "Co2Transpiration",
    "HeatStressOnGrain",
    "HeatStressOnLeafSenescence",
    "Irradiation",
    "LeafDynamics",
    "NutrientDemand",
    "Partitioning",
    "Phenology",
    "Photosynthesis",
    "PotentialEvapoTranspiration",
    "RootDynamics",
    "SoilNutrients",
    "StemDynamics",
    "StressFactors",
    "WaterBalance",
]
