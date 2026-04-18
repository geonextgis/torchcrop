"""Biophysical process modules."""

from torchcrop.processes.astro import Astro
from torchcrop.processes.evapotranspiration import PotentialEvapoTranspiration
from torchcrop.processes.irradiation import Irradiation
from torchcrop.processes.leaf_dynamics import LeafDynamics
from torchcrop.processes.nutrient_demand import NutrientDemand
from torchcrop.processes.partitioning import Partitioning
from torchcrop.processes.phenology import Phenology
from torchcrop.processes.photosynthesis import Photosynthesis
from torchcrop.processes.root_dynamics import RootDynamics
from torchcrop.processes.stress import StressFactors
from torchcrop.processes.water_balance import WaterBalance

__all__ = [
    "Astro",
    "Irradiation",
    "LeafDynamics",
    "NutrientDemand",
    "Partitioning",
    "Phenology",
    "Photosynthesis",
    "PotentialEvapoTranspiration",
    "RootDynamics",
    "StressFactors",
    "WaterBalance",
]
