"""
Pages package for the BigFoot dashboard
"""

# Import all page modules to make them available
from . import overview
from . import geographic
from . import temporal
from . import machine_learning
from . import anomaly
from . import recommendations
from . import explorer
from . import about

__all__ = [
    'overview',
    'geographic', 
    'temporal',
    'machine_learning',
    'anomaly',
    'recommendations',
    'explorer',
    'about'
]