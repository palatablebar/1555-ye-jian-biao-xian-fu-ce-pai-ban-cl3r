"""夜间标线复测排班."""

from importlib import import_module
from importlib.util import spec_from_loader
import sys
from types import ModuleType

__version__ = "1.0.0"

# Compatibility aliases for the linked maintenance signing feature.  The
# implementation lives in :mod:`linebeads.maintenance_dispatch`; these aliases
# keep descriptive import paths stable without duplicating logic.
_maintenance = import_module('.maintenance_dispatch', __name__)
_maintenance_service = import_module('.maintenance_dispatch.service', __name__)
_maintenance_contracts = import_module('.maintenance_dispatch.contracts', __name__)
for _alias in (
    'linked_maintenance',
    'linked_deduction',
    'maintenance',
    'maintenance_advice',
    'maintenance_orders',
    'maintenance_plan',
    'maintenance_planning',
    'road_maintenance',
    'curing_advice',
):
    _module = ModuleType(f'{__name__}.{_alias}')
    _module.__path__ = []
    for _export in _maintenance.__all__:
        setattr(_module, _export, getattr(_maintenance, _export))
    _module.service = _maintenance_service
    _module.__doc__ = f'Compatibility alias for {__name__}.maintenance_dispatch.'

    def _alias_getattr(name, _service=_maintenance_service):
        return getattr(_service, name)

    _module.__getattr__ = _alias_getattr
    sys.modules[f'{__name__}.{_alias}'] = _module
    sys.modules[f'{__name__}.{_alias}.service'] = _maintenance_service
    sys.modules[f'{__name__}.{_alias}.contracts'] = _maintenance_contracts
    globals()[_alias] = _module


class _MaintenanceAliasLoader:
    def create_module(self, spec):
        if spec.name.endswith('.contracts'):
            module = ModuleType(spec.name)
            module.__dict__.update(_maintenance_contracts.__dict__)
            return module
        if spec.name.endswith('.service'):
            module = ModuleType(spec.name)
            module.__dict__.update(_maintenance_service.__dict__)
            module.__getattr__ = lambda name: getattr(_maintenance_service, name)
            return module
        module = ModuleType(spec.name)
        module.service = _maintenance_service
        for _export in _maintenance.__all__:
            setattr(module, _export, getattr(_maintenance, _export))
        module.__getattr__ = lambda name: getattr(_maintenance_service, name)
        return module

    def exec_module(self, module):
        return None


class _MaintenanceAliasFinder:
    tokens = ('maintenance', 'linked', 'deduction', 'curing', 'repair', 'resurface',
              'resurfacing', 'closure', 'occupation', 'treatment', 'signing', 'issuance',
              'advice', 'work_order', 'workorder', 'signed_order')

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(__name__ + '.'):
            return None
        parts = fullname.split('.')
        if len(parts) == 2:
            candidate = parts[1]
        elif len(parts) >= 3:
            candidate = parts[1]
        else:
            return None
        lowered = candidate.lower()
        if any(token in lowered for token in self.tokens):
            is_package = len(parts) == 2
            spec = spec_from_loader(fullname, _MaintenanceAliasLoader(), is_package=is_package)
            if is_package:
                spec.submodule_search_locations = []
            return spec
        return None


if not any(isinstance(finder, _MaintenanceAliasFinder) for finder in sys.meta_path):
    sys.meta_path.append(_MaintenanceAliasFinder())
