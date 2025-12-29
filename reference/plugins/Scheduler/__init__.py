"""
Scheduling package
==================
High-level import shortcuts so GUI code can write:

    from Scheduling import (
        AssignmentPassenger, PassengerTimeScheduler,
        RegularFlightScheduler, simulate_ground_operations,
        OptimizeFATO, OptimizeCapacity,
    )
"""
from importlib import import_module
from pathlib import Path
import pkgutil

# expose everything from Scheduling.Functions.*
_pkg_path = Path(__file__).with_suffix('')
for mod in pkgutil.iter_modules([_pkg_path / "Functions"]):
    import_module(f"Scheduling.Functions.{mod.name}")

from .Functions.AssignmentPassenger       import AssignmentPassenger
from .Functions.PassengerTimeScheduler    import DemandProfile, PassengerTimeScheduler
from .Functions.Scheduling_Optimized      import (
    RegularFlightScheduler,
    simulate_ground_operations,
)
# from .Notuse.Optimize_FATO             import sweep_runway_capacity as OptimizeFATO
# from .Notuse.Optimize_Capacity         import max_ready_rate, min_fato_capacity as OptimizeCapacity

__all__ = [
    "AssignmentPassenger", "DemandProfile", "PassengerTimeScheduler",
    "RegularFlightScheduler", "simulate_ground_operations",
    "OptimizeFATO", "OptimizeCapacity",
]
