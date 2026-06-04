"""torchcrop — differentiable Lintul5 crop-growth model in PyTorch.

Top-level package exposing the public API of torchcrop.

References:
    Wolf, J. (2012). User guide for LINTUL5. Wageningen UR.
"""

__author__ = """Krishnagopal Halder"""
__email__ = "geonextgis@gmail.com"
__version__ = "1.0.0"

from torchcrop.config import RunConfig
from torchcrop.drivers.weather import WeatherDriver
from torchcrop.engine import SimulationEngine
from torchcrop.model import Lintul5Model, ModelOutput
from torchcrop.parameters.crop_params import CropParameters, available_crops
from torchcrop.parameters.site_params import SiteParameters
from torchcrop.parameters.soil_params import SoilParameters
from torchcrop.states.model_state import DiagnosticState, ModelState

__all__ = [
    "CropParameters",
    "DiagnosticState",
    "Lintul5Model",
    "available_crops",
    "ModelOutput",
    "ModelState",
    "RunConfig",
    "SimulationEngine",
    "SiteParameters",
    "SoilParameters",
    "WeatherDriver",
    "__version__",
]
