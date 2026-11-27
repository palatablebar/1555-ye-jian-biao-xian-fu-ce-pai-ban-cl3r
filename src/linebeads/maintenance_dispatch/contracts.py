"""Contract types for linked maintenance issuance.

The functional implementation currently uses JSON-friendly :class:`Flex`
records.  Those records also expose attributes, while these aliases give callers
and tests a stable contracts module.
"""

from .service import (
    Bounds,
    Calibration,
    Flex,
    MaintenanceOrder,
    MaintenancePlan,
    MaintenanceTicket,
    ReductionModel,
    SignedMaintenanceOrder,
)

__all__ = (
    'Bounds', 'Calibration', 'Flex', 'MaintenanceOrder', 'MaintenancePlan',
    'MaintenanceTicket', 'ReductionModel', 'SignedMaintenanceOrder',
)
