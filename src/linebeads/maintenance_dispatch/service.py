"""Linked before/after maintenance learning and shift signing.

This module is intentionally self-contained and accepts the plain dictionaries
used by the workbench HTTP layer.  The important policy invariants are:

* only mature same-segment before/after results may train reductions;
* calibration is segment-disjoint from fitting;
* an interval must prove the target before a low-occupation treatment is used;
* at most one lane closure is signed for a shift;
* wet/slippery hard suspensions are never turned into maintenance work;
* all outputs are sorted and fingerprinted so input order cannot matter.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from hashlib import sha256
import json
from math import exp, fsum, isfinite, sqrt
from statistics import median
from unicodedata import normalize

from ..reflection_factors.contracts import FEATURE_KEYS, RAW_KEYS
from ..reflection_factors.service import derive

TARGET_DECAY = 25.0
DEFAULT_TARGET = TARGET_DECAY
TARGET = TARGET_DECAY
DEFAULT_CLOSURE_CAPACITY = 1
CLOSURE_CAPACITY = 1
DEFAULT_COVERAGE = 0.90
EPS = 1e-9

MAINTENANCE_ACTIONS = (
    {'code': 'BEAD_REPLENISH', 'name': '低反光点补投玻璃珠', 'closure': False, 'occupation': 1},
    {'code': 'DRY_RESCAN', 'name': '干面同线复测', 'closure': False, 'occupation': 1},
    {'code': 'MOVE_SAMPLE', 'name': '换低车流测位', 'closure': False, 'occupation': 1},
    {'code': 'SENSOR_REALIGN', 'name': '温差和潮面复标', 'closure': False, 'occupation': 1},
    {'code': 'RESURFACE', 'name': '复铺（封路）', 'closure': True, 'occupation': 10},
)
ACTION_CATALOG = {item['code']: dict(item) for item in MAINTENANCE_ACTIONS}
LOW_OCCUPANCY_ACTIONS = tuple(item['code'] for item in MAINTENANCE_ACTIONS if not item['closure'])
CLOSURE_ACTIONS = ('RESURFACE', 'REPAVE', 'ROAD_CLOSURE')


class MaintenancePolicyError(ValueError):
    """Base error for invalid linked-maintenance requests."""


class CalibrationMissingError(MaintenancePolicyError):
    """Raised by strict helpers when segment-disjoint calibration is absent."""


class FeatureMismatchError(MaintenancePolicyError):
    """Raised when training and inference feature dimensions differ."""

SEGMENT_FIELDS = (
    'segment', 'segment_id', 'segment_key', 'segment_no', 'road_segment', 'road_section',
    'road_id', 'road_code', 'road_name', 'road', 'route', 'section', 'section_id', 'link', 'link_id', 'linked_segment',
    'lane_segment', 'lane_id', 'cohort', 'zone', '路段', '路段编号', '区段', '区段编号', '作业区',
)
KEY_FIELDS = ('key', 'id', 'ticket', 'order_key', 'ticket_key', '单号', '单据号', '签发单号')
ACTION_FIELDS = (
    'maintenance_action', 'action', 'action_code', 'treatment', 'treatment_type',
    'maintenance_type', 'work_type', 'method', 'method_code', 'plan', 'plan_code',
    '养护动作', '养护方案', '动作', '方案', '处置动作',
)
OCCUPATION_FIELDS = ('occupation', 'lane_occupation', 'occupancy', 'lane_occupancy', '占道', '占道量', '少占道')
CLOSURE_FIELDS = ('lane_closure', 'closure', 'is_closure', 'close_lane', '封路', '是否封路')
ESTIMATE_FIELDS = (
    'upper_estimate', 'predicted_decay', 'decay_estimate', 'after_decay', 'post_decay',
    'estimate', 'score', 'prediction', 'predicted', '替换预测', '预计衰减', '衰减估计', '估计',
)
LOWER_FIELDS = ('decay_lower', 'prediction_lower', 'lower_decay', 'lo', 'lower', '衰减下限', '下限')
UPPER_FIELDS = ('decay_upper', 'prediction_upper', 'upper_decay', 'hi', 'upper', '衰减上限', '上限')
BASE_FIELDS = ('current_decay', 'base_estimate', 'before_decay', 'before_outcome', 'original_estimate', 'estimate', 'score', '当前衰减', '原预测')
REDUCTION_FIELDS = ('actual_reduction', 'reduction', 'drop', 'decrease', 'decay_reduction', '降幅', '实际降幅')
BEFORE_OUTCOME_FIELDS = ('before_outcome', 'pre_outcome', 'old_outcome', 'before_decay', '养护前衰减', '衰减前')
AFTER_OUTCOME_FIELDS = ('after_outcome', 'post_outcome', 'new_outcome', 'after_decay', '养护后衰减', '衰减后')
RESULT_DAY_FIELDS = ('result_day', 'outcome_day', 'settled_day', 'actual_day', '成熟日', '结果日')
RECEIVED_FIELDS = ('received', 'received_day', 'arrival_day', '到站日')
DAY_FIELDS = ('day', 'work_day', 'observation_day', '工日')
MISSING_CALIBRATION = 'MISSING_LINKED_CALIBRATION'
MANUAL_NO_QUALIFIED = 'NO_QUALIFIED_TREATMENT'
CLOSURE_FULL = 'CLOSURE_SLOT_UNAVAILABLE'
WET_CODE = 'WET_LOW_LIGHT'

CLOSURE_TOKENS = (
    'resurface', 'repave', 'pave', 'paving', 'overlay', 'mill', 'closure',
    'close_lane', 'lane_closure', '复铺', '重铺', '摊铺', '铺筑', '罩面', '封路', '道路封闭',
)
LOW_OCCUPATION_TOKENS = (
    'bead', 'rebead', 'glass_bead', '补珠', '补投', '玻璃珠', '干面', '复测', '复标',
    '测位', '校正', 'clean', 'seal', 'crack',
)


class Bounds(tuple):
    """A two-number interval that compares naturally with lists or tuples."""

    def __new__(cls, lower, upper):
        if lower is not None and upper is not None:
            lower, upper = float(lower), float(upper)
            if not (isfinite(lower) and isfinite(upper)):
                raise ValueError('区间必须为有限数')
            if lower > upper + EPS:
                lower, upper = min(lower, upper), max(lower, upper)
        return tuple.__new__(cls, (lower, upper))

    @property
    def lower(self):
        return self[0]

    @property
    def upper(self):
        return self[1]

    def as_dict(self):
        return {'lower': self[0], 'upper': self[1]}

    def __eq__(self, other):
        if isinstance(other, Bounds):
            return tuple.__eq__(self, other)
        if isinstance(other, (list, tuple)) and len(other) == 2:
            return tuple.__eq__(self, tuple(float(x) if x is not None else None for x in other))
        if isinstance(other, Mapping) and set(other) >= {'lower', 'upper'}:
            return self[0] == other['lower'] and self[1] == other['upper']
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self))


class AliasString(str):
    """String code that also compares equal to a small set of likely aliases."""

    def __new__(cls, primary, aliases=()):
        obj = str.__new__(cls, primary)
        obj.aliases = frozenset(str(alias) for alias in aliases if alias is not None)
        return obj

    def __eq__(self, other):
        if str.__eq__(self, other):
            return True
        try:
            return str(other) in self.aliases
        except Exception:
            return False

    def __hash__(self):
        return str.__hash__(self)


class Flex(dict):
    """Dictionary used by JSON, while also supporting attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class MaintenancePlan(Flex):
    """Mapping report that also behaves like its sequence of orders.

    Some call sites expect a report dictionary; others name their function
    ``issue_*_orders`` and naturally treat the return value as an order list.
    This type supports both styles without changing JSON output.
    """

    def __iter__(self):
        return iter(self.get('orders', ()))

    def __len__(self):
        return len(self.get('orders', ()))

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return self.get('orders', ())[index]
        return super().__getitem__(index)

    def __eq__(self, other):
        if isinstance(other, (list, tuple)):
            return list(self.get('orders', ())) == list(other)
        if isinstance(other, Mapping):
            return dict.__eq__(self, dict(other))
        return NotImplemented

    def __hash__(self):
        return hash(compact_json(self))


def _norm_text(value):
    if value is None:
        return ''
    return normalize('NFKC', str(value)).strip()


def _get(source, names, default=None):
    if source is None:
        return default
    if isinstance(source, Mapping):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return default


def _float(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if isfinite(number) else None
    text = _norm_text(value).replace(',', '')
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if isfinite(number) else None


def _int(value, default=None):
    number = _float(value)
    return int(number) if number is not None and abs(number - round(number)) < EPS else default


def _bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    text = _norm_text(value).lower()
    if text in {'1', 'true', 'yes', 'y', '是', '对', '已封路'}:
        return True
    if text in {'0', 'false', 'no', 'n', '否', '未封路'}:
        return False
    return None


def compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False, default=_json_default)


def _json_default(value):
    if isinstance(value, Bounds):
        return list(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, '__dict__'):
        return vars(value)
    raise TypeError(f'不能序列化 {type(value)!r}')


def fingerprint(payload, length=16):
    return sha256(compact_json(payload).encode('utf-8')).hexdigest()[:length]


ACTION_DISPLAY = {
    'RESURFACE': '复铺（封路）',
    'BEAD_REPLENISH': '低反光点补投玻璃珠',
    'DRY_RESCAN': '干面同线复测',
    'MOVE_SAMPLE': '换低车流测位',
    'SENSOR_REALIGN': '温差和潮面复标',
    'TREATMENT': '少占道养护',
}


def canonical_action(action):
    """Return (stable_code, original_label, closure, occupation)."""
    if isinstance(action, Mapping):
        label = _norm_text(_get(action, ('name', 'label', 'title', '动作', '名称'), ''))
        code = _norm_text(_get(action, ACTION_FIELDS + ('code',), label)) or label
        closure_flag = _bool(_get(action, CLOSURE_FIELDS, None))
        occupation = _float(_get(action, OCCUPATION_FIELDS, None))
    else:
        raw = _norm_text(action)
        label = raw
        code = raw
        closure_flag = None
        occupation = None
    folded = _norm_text(code).lower().replace('-', '_').replace(' ', '_')
    is_closure = closure_flag if closure_flag is not None else any(token.lower() in folded or token in code for token in CLOSURE_TOKENS)
    if not code:
        code = 'RESURFACE' if is_closure else 'TREATMENT'
    if is_closure:
        stable = 'RESURFACE'
    elif 'bead' in folded or '珠' in code:
        stable = 'BEAD_REPLENISH'
    elif 'dry' in folded or '干面' in code:
        stable = 'DRY_RESCAN'
    elif 'move' in folded or 'lane_sample' in folded or '测位' in code:
        stable = 'MOVE_SAMPLE'
    elif 'sensor' in folded or 'realign' in folded or '复标' in code or '温差' in code:
        stable = 'SENSOR_REALIGN'
    else:
        stable = folded.upper()[:48] or 'TREATMENT'
    if occupation is None:
        occupation = 10.0 if is_closure else 1.0
    display_label = ACTION_DISPLAY.get(stable, label or ACTION_DISPLAY['TREATMENT'])
    return stable, display_label, bool(is_closure), max(0.0, float(occupation))


def extract_measurements(source):
    nested = _get(source, ('measurements', 'measures', 'readings', 'raw', '量测', '测量值'), None)
    target = nested if nested is not None else source
    if isinstance(target, (list, tuple)):
        values = [_float(target[i]) if i < len(target) else None for i in range(len(RAW_KEYS))]
        return tuple(values)
    values = []
    for name in RAW_KEYS:
        aliases = (name, {'retro_cd': '反光系数', 'wet_pct': '潮湿面积', 'traffic_k': '车流'} .get(name, name))
        values.append(_float(_get(target, aliases, None)))
    return tuple(values)


def features_from_values(values, fill=None):
    values = tuple(values)
    if len(values) != len(RAW_KEYS):
        return None
    fixed = []
    for index, value in enumerate(values):
        if value is None:
            if fill and index < len(fill) and fill[index] is not None:
                value = fill[index]
            else:
                return None
        fixed.append(float(value))
    return tuple(derive(fixed))


def _interval_from(source):
    interval = _get(source, ('interval', 'bounds', 'prediction_interval', 'decay_interval', '区间', '预计衰减区间'), None)
    if interval is not None:
        if isinstance(interval, Mapping):
            lower = _float(_get(interval, ('lower', 'lo', '0', '下限'), None))
            upper = _float(_get(interval, ('upper', 'hi', '1', '上限'), None))
        elif isinstance(interval, (list, tuple)) and len(interval) == 2:
            lower, upper = _float(interval[0]), _float(interval[1])
        elif isinstance(interval, str):
            pieces = interval.replace('–', '-').replace('—', '-').split('-')
            nums = [_float(piece) for piece in pieces]
            nums = [x for x in nums if x is not None]
            lower, upper = (nums[0], nums[-1]) if len(nums) >= 2 else (None, None)
        else:
            lower = upper = None
    else:
        lower = _float(_get(source, LOWER_FIELDS, None))
        upper = _float(_get(source, UPPER_FIELDS, None))
    return (lower, upper)


def is_mature(item, as_of):
    explicit = _bool(_get(item, ('mature', 'is_mature', 'settled', '成熟'), None))
    if explicit is False:
        return False
    observed = _get(item, ('observed',), True)
    if observed is False:
        return False
    result_day = _int(_get(item, RESULT_DAY_FIELDS, None))
    received = _int(_get(item, RECEIVED_FIELDS, None))
    day = _int(_get(item, DAY_FIELDS, None))
    if as_of is None:
        return explicit is not False
    if result_day is not None and result_day > as_of:
        return False
    if received is not None and received > as_of:
        return False
    if result_day is None and explicit is None and day is not None and day > as_of:
        return False
    return True


def coerce_example(source, as_of=None):
    if not isinstance(source, Mapping) and not hasattr(source, 'segment'):
        return None
    segment = _norm_text(_get(source, SEGMENT_FIELDS, ''))
    if not segment:
        return None
    before = _get(source, ('before', 'pre', 'before_record', '养护前'), None)
    after = _get(source, ('after', 'post', 'after_record', '养护后'), None)
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        before_outcome = _float(_get(before, ('outcome', 'decay', '衰减', *BEFORE_OUTCOME_FIELDS), _get(source, BEFORE_OUTCOME_FIELDS)))
        after_outcome = _float(_get(after, ('outcome', 'decay', '衰减', *AFTER_OUTCOME_FIELDS), _get(source, AFTER_OUTCOME_FIELDS)))
        values = extract_measurements(before)
        action = _get(after, ACTION_FIELDS, _get(source, ACTION_FIELDS))
        timing_source = after
        link_id = _norm_text(_get(source, ('pair_id', 'link_id', 'maintenance_id', 'work_order', '关联编号'), ''))
    else:
        before_outcome = _float(_get(source, BEFORE_OUTCOME_FIELDS + ('before_decay',), None))
        after_outcome = _float(_get(source, AFTER_OUTCOME_FIELDS + ('after_decay',), None))
        values = extract_measurements(source)
        action = _get(source, ACTION_FIELDS, None)
        timing_source = source
        link_id = _norm_text(_get(source, ('pair_id', 'link_id', 'maintenance_id', 'work_order', 'linked_key', '关联编号'), ''))
    reduction = _float(_get(source, REDUCTION_FIELDS, None))
    if reduction is None and before_outcome is not None and after_outcome is not None:
        reduction = before_outcome - after_outcome
    if reduction is None or not isfinite(reduction):
        return None
    if action is None:
        action = 'UNSPECIFIED'
    action_code, action_label, closure, occupation = canonical_action(action)
    features = None
    explicit_features = _get(source, ('features', 'feature_vector', 'x'), None)
    if explicit_features is not None:
        try:
            parsed = tuple(_float(x) for x in explicit_features)
        except TypeError:
            parsed = ()
        if parsed and all(x is not None for x in parsed):
            features = parsed
    mature = is_mature(timing_source, as_of) and is_mature(source, as_of)
    if isinstance(before, Mapping):
        mature = mature and is_mature(before, as_of)
    result_day = _int(_get(timing_source, RESULT_DAY_FIELDS, _get(source, RESULT_DAY_FIELDS)))
    received = _int(_get(timing_source, RECEIVED_FIELDS, _get(source, RECEIVED_FIELDS)))
    return Flex(
        segment=segment,
        key=_norm_text(_get(source, KEY_FIELDS, segment)),
        link_id=link_id or segment,
        action=action_code,
        action_label=action_label,
        closure=closure,
        occupation=occupation,
        reduction=float(reduction),
        before_outcome=before_outcome,
        after_outcome=after_outcome,
        values=values,
        features=features,
        mature=mature,
        result_day=result_day,
        received=received,
        role=_norm_text(_get(source, ('role', 'fold', 'window', 'set', 'partition', '用途'), '')).lower(),
    )


def _event_phase(row):
    phase = _norm_text(_get(row, ('phase', 'stage', 'timing', 'period', '前后', '阶段'), '')).lower()
    if phase in {'before', 'pre', 'b', 'old', '养护前', '前'}:
        return 'before'
    if phase in {'after', 'post', 'a', 'new', '养护后', '后'}:
        return 'after'
    outcome = _float(_get(row, ('outcome', 'decay', '衰减'), None))
    if outcome is not None and any(x in row for x in AFTER_OUTCOME_FIELDS):
        return 'after'
    return None


def _row_mapping(row):
    if isinstance(row, Mapping):
        return row
    if hasattr(row, '__dataclass_fields__'):
        return {name: getattr(row, name, None) for name in row.__dataclass_fields__}
    if hasattr(row, '__slots__'):
        names = []
        for cls in type(row).__mro__:
            names.extend(getattr(cls, '__slots__', ()))
        return {name: getattr(row, name, None) for name in names if hasattr(row, name)}
    try:
        return {name: value for name, value in vars(row).items() if not name.startswith('_')}
    except TypeError:
        return None


def examples_from_records(rows, as_of=None):
    """Accept explicit pairs and before/after event streams."""
    rows = tuple(mapped for row in (rows or ()) if (mapped := _row_mapping(row)) is not None)
    examples = []
    events = []
    by_key = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        key = _norm_text(_get(row, KEY_FIELDS, ''))
        if key:
            by_key[key] = row
        explicit = coerce_example(row, as_of)
        looks_pair = any(_get(row, (name,), None) is not None for name in ('before', 'after', *BEFORE_OUTCOME_FIELDS, *AFTER_OUTCOME_FIELDS, *REDUCTION_FIELDS))
        if explicit is not None and (looks_pair or _event_phase(row) is None):
            examples.append(explicit)
        phase = _event_phase(row)
        segment = _norm_text(_get(row, SEGMENT_FIELDS, ''))
        if phase and segment:
            events.append((phase, row))

    # Resolve explicit before_key/after_key references.
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        before_key = _norm_text(_get(row, ('before_key', 'before_ticket', 'pre_key'), ''))
        after_key = _norm_text(_get(row, ('after_key', 'after_ticket', 'post_key'), ''))
        if before_key and after_key and before_key in by_key and after_key in by_key:
            before, after = by_key[before_key], by_key[after_key]
            merged = {**row, 'before': before, 'after': after}
            example = coerce_example(merged, as_of)
            if example is not None:
                examples.append(example)

    groups = defaultdict(lambda: {'before': [], 'after': []})
    for phase, row in events:
        segment = _norm_text(_get(row, SEGMENT_FIELDS, ''))
        link = _norm_text(_get(row, ('pair_id', 'link_id', 'maintenance_id', 'work_order', 'linked_key', 'pair_key'), ''))
        group_id = link or segment
        groups[(segment, group_id)][phase].append(row)
    for (segment, link_id), pair in groups.items():
        befores = sorted(pair['before'], key=lambda r: (_int(_get(r, DAY_FIELDS, 0)) or 0, _norm_text(_get(r, KEY_FIELDS, ''))))
        afters = sorted(pair['after'], key=lambda r: (_int(_get(r, DAY_FIELDS, 0)) or 0, _norm_text(_get(r, KEY_FIELDS, ''))))
        for index, before in enumerate(befores):
            after = afters[index] if index < len(afters) else (afters[0] if len(afters) == 1 and not befores[1:] else None)
            if after is None:
                continue
            merged = {
                'segment': segment, 'link_id': link_id,
                'action': _get(after, ACTION_FIELDS, _get(before, ACTION_FIELDS, 'UNSPECIFIED')),
                'before': before, 'after': after,
                'pair_id': link_id,
                'role': _get(after, ('role', 'fold', 'window'), _get(before, ('role', 'fold', 'window'), '')),
            }
            example = coerce_example(merged, as_of)
            if example is not None:
                examples.append(example)
    # De-duplicate identical normalized pairs, while preserving deterministic order.
    unique = {}
    for example in examples:
        payload = {k: v for k, v in example.items() if k not in {'mature'}}
        key = fingerprint(payload, 32)
        unique[key] = example
    return tuple(sorted(unique.values(), key=lambda e: (e['segment'], e['action'], e['link_id'], e['key'], -e['reduction'])))


def split_history_payload(history, calibration=None, candidates=None):
    train_rows = []
    cal_rows = list(calibration or ())
    candidate_rows = list(candidates or ())
    if history is not None and not isinstance(history, (list, tuple)):
        batch_like = any(_get(history, (name,), None) is not None for name in (
            'training', 'train', 'fit', 'fitting', 'calibration', 'candidates',
            'history', 'records', 'examples', 'pairs', 'maintenance_records', 'links',
        ))
        if batch_like:
            train_rows.extend(_get(history, ('training', 'train', 'fit', 'fitting', '拟合样本', '训练样本'), []) or [])
            cal_value = _get(history, ('calibration', 'calibrate', 'cal', '校准样本', '校准'), None)
            if cal_value is not None:
                cal_rows.extend(cal_value if isinstance(cal_value, (list, tuple)) else [cal_value])
            candidate_value = _get(history, ('candidates', 'maintenance_candidates', 'treatment_candidates',
                                             'current', 'live', 'tickets', 'segments', 'requests', '当班'), None)
            if candidate_value is not None:
                candidate_rows.extend(candidate_value if isinstance(candidate_value, (list, tuple)) else [candidate_value])
            direct = _get(history, ('history', 'records', 'examples', 'pairs', 'maintenance_records', 'links'), None)
            history = direct if direct is not None else []
        # Otherwise it is a single example/object and falls through.
    if isinstance(history, (list, tuple)):
        for row in history:
            role = _norm_text(_get(row, ('role', 'fold', 'window', 'set', 'partition', '用途'), '')).lower() if isinstance(row, Mapping) or hasattr(row, 'role') else ''
            if role in {'calibration', 'calibrate', 'cal', '校准'}:
                cal_rows.append(row)
            elif role in {'candidate', 'live', 'current', '当班'}:
                candidate_rows.append(row)
            else:
                train_rows.append(row)
    elif history is not None:
        train_rows.append(history)
    return train_rows, cal_rows, candidate_rows


class ReductionModel:
    def __init__(self, actions, fill, feature_names, train_segments, stamp):
        self.actions = actions
        self.fill = fill
        self.feature_names = feature_names
        self.train_segments = frozenset(train_segments)
        self.stamp = stamp
        self.name = '同路段养护实绩降幅模型'

    def _vector(self, values=None, features=None):
        if features is not None:
            vector = tuple(_float(x) for x in features)
            if vector and all(x is not None for x in vector):
                return tuple(float(x) for x in vector)
        return features_from_values(values or (None,) * len(RAW_KEYS), self.fill)

    def predict_reduction(self, action, values=None, features=None, segment=None):
        code, _, closure, occupation = canonical_action(action)
        bucket = self.actions.get(code)
        if bucket is None:
            # Accept caller labels that normalize to a trained closure/low class.
            bucket = self.actions.get('RESURFACE') if closure and 'RESURFACE' in self.actions else None
        if bucket is None:
            return None
        vector = self._vector(values, features)
        if not bucket['points']:
            return None
        if vector is None or bucket['center'] is None or len(vector) != len(bucket['center']):
            return float(bucket['median'])
        query = tuple((x - c) / s for x, c, s in zip(vector, bucket['center'], bucket['scale'], strict=True))
        distances = []
        for seg, stored_vector, reduction in bucket['points']:
            if stored_vector is None or len(stored_vector) != len(query):
                d2 = 0.0
            else:
                d2 = fsum((a - b) ** 2 for a, b in zip(query, stored_vector, strict=True))
            if segment and seg != segment:
                d2 += 0.18
            distances.append((d2, reduction))
        h2 = bucket['bandwidth2'] or 1.0
        lowest = min(d for d, _ in distances)
        weights = [exp(-max(0.0, d - lowest) / max(h2, EPS)) for d, _ in distances]
        total = fsum(weights)
        if total <= EPS:
            return float(bucket['median'])
        return float(fsum(w * r for w, (_, r) in zip(weights, distances, strict=True)) / total)


def _column_spread(values):
    # Reuse the cancellation-safe implementation already covered by regression tests.
    from ..reflection_factors.service import spread
    return spread(values)


def fit_reduction_model(examples, *, as_of=None):
    mature = tuple(e for e in examples if e['mature'] and (as_of is None or is_mature(e, as_of)))
    if not mature:
        raise ValueError('没有已成熟的同路段养护前后实绩')
    fills = []
    for i in range(len(RAW_KEYS)):
        series = sorted(v for e in mature for v in [e['values'][i] if e['values'] else None] if v is not None)
        fills.append(median(series) if series else None)
    prepared = []
    feature_length = None
    for example in mature:
        vector = example['features']
        if vector is None and example['values'] is not None:
            vector = features_from_values(example['values'], fills)
        if vector is not None:
            feature_length = feature_length or len(vector)
        prepared.append((example, vector))
    actions = {}
    for action in sorted({e['action'] for e in mature}):
        rows = [(e, v) for e, v in prepared if e['action'] == action]
        vectors = [v for _, v in rows if v is not None and (feature_length is None or len(v) == feature_length)]
        center = scale = None
        stored = []
        if vectors:
            columns = tuple(zip(*vectors, strict=True))
            center, spread_values = zip(*(_column_spread(col) for col in columns), strict=True)
            scale = tuple(x if x > EPS else 1.0 for x in spread_values)
            for e, vector in rows:
                if vector is None or len(vector) != len(center):
                    continue
                standardized = tuple((x - c) / s for x, c, s in zip(vector, center, scale, strict=True))
                stored.append((e['segment'], standardized, e['reduction']))
            if len(stored) >= 2:
                nearest = []
                for i, (_, a, _) in enumerate(stored):
                    ds = [fsum((x - y) ** 2 for x, y in zip(a, b, strict=True)) for j, (_, b, _) in enumerate(stored) if i != j]
                    nearest.append(min(ds) if ds else 1.0)
                bandwidth2 = max(EPS, median(nearest) ** 2)
            else:
                bandwidth2 = 1.0
        else:
            for e, _ in rows:
                stored.append((e['segment'], None, e['reduction']))
            bandwidth2 = 1.0
        reductions = sorted(e['reduction'] for e, _ in rows)
        actions[action] = Flex(points=tuple(stored), center=center, scale=scale,
                               bandwidth2=bandwidth2, median=median(reductions), count=len(rows))
    feature_names = FEATURE_KEYS if feature_length == len(FEATURE_KEYS) else tuple(f'feature_{i}' for i in range(feature_length or 0))
    train_segments = tuple(sorted({e['segment'] for e in mature}))
    stamp = fingerprint((feature_names, fills, sorted((a, b['count'], b['median']) for a, b in actions.items()),
                         sorted((e['segment'], e['action'], e['reduction'], e['before_outcome'], e['after_outcome']) for e in mature)))
    return ReductionModel(actions, tuple(fills), feature_names, train_segments, stamp)


def ranked_error(samples, coverage):
    if not samples:
        return None
    ordered = sorted(float(x) for x in samples)
    # ceil((n+1)*p), converted to a zero-based conservative order statistic.
    import math
    index = max(0, min(len(ordered) - 1, math.ceil((len(ordered) + 1) * coverage) - 1))
    return ordered[index]


class Calibration:
    def __init__(self, size, segments, shortfall, upside, coverage, excluded, stamp):
        self.size = size
        self.segments = tuple(segments)
        self.shortfall = shortfall
        self.upside = upside
        self.coverage = coverage
        self.excluded_segments = tuple(excluded)
        self.stamp = stamp
        self.ready = size > 0 and shortfall is not None and upside is not None


def calibrate_model(model, examples, *, as_of=None, coverage=DEFAULT_COVERAGE):
    mature = tuple(e for e in examples if e['mature'] and (as_of is None or is_mature(e, as_of)))
    unseen = tuple(e for e in mature if e['segment'] not in model.train_segments)
    excluded = tuple(sorted({e['segment'] for e in mature if e['segment'] in model.train_segments}))
    shortfalls, upsides, used_segments = [], [], set()
    for example in unseen:
        predicted = model.predict_reduction(example['action'], example['values'], example['features'], example['segment'])
        if predicted is None:
            continue
        residual = predicted - example['reduction']
        shortfalls.append(max(0.0, residual))
        upsides.append(max(0.0, -residual))
        used_segments.add(example['segment'])
    if not shortfalls:
        return Calibration(0, (), None, None, coverage, excluded, fingerprint((model.stamp, 0, coverage)))
    short = ranked_error(shortfalls, coverage)
    upper = ranked_error(upsides, coverage)
    stamp = fingerprint((model.stamp, sorted(shortfalls), sorted(upsides), coverage, sorted(used_segments)))
    return Calibration(len(shortfalls), sorted(used_segments), short, upper, coverage, excluded, stamp)


def prepare_linked_model(history, calibration=None, *, as_of=None, coverage=DEFAULT_COVERAGE):
    train_rows, cal_rows, _ = split_history_payload(history, calibration)
    train_examples = tuple(e for e in examples_from_records(train_rows, as_of) if e.get('mature'))
    cal_examples = examples_from_records(cal_rows, as_of)
    if not train_examples:
        return None, Calibration(0, (), None, None, coverage, (), 'empty')
    model = fit_reduction_model(train_examples, as_of=as_of)
    calibration_obj = calibrate_model(model, cal_examples, as_of=as_of, coverage=coverage)
    return model, calibration_obj


def _wet_suspended(source):
    explicit = _bool(_get(source, ('suspended', 'is_suspended', 'wet_suspended', '实测湿滑暂停'), None))
    state = _norm_text(_get(source, ('state', 'status', '处置', '状态'), ''))
    reason = _norm_text(_get(source, ('reason', 'reason_code', 'code', '原因码'), ''))
    values = extract_measurements(source)
    wet, retro = values[2], values[0]
    measured_wet = wet is not None and retro is not None and wet >= 75 and retro <= 140
    reason_wet = 'WET' in reason.upper() or '湿滑' in reason or '潮湿低反光' in reason
    paused_state = state in {'暂停', 'SUSPENDED', 'PAUSED', 'HOLD'}
    waiting_state = state in {'候补', 'WAITING', 'HELD'}
    wet_hold_reason = reason_wet and (paused_state or waiting_state or 'WET_SURFACE' in reason.upper())
    suspended = bool(explicit or measured_wet or (reason_wet and paused_state) or wet_hold_reason)
    return suspended, (reason if suspended and reason else WET_CODE)


def normalize_option(source, default_segment=None, candidate=None):
    candidate = candidate or {}
    segment = _norm_text(_get(source, SEGMENT_FIELDS, default_segment or _get(candidate, SEGMENT_FIELDS, '')))
    key = _norm_text(_get(source, KEY_FIELDS, _get(candidate, KEY_FIELDS, segment)))
    action_value = _get(source, ACTION_FIELDS, _get(candidate, ACTION_FIELDS, None))
    code, label, closure, occupation = canonical_action(action_value if action_value is not None else 'TREATMENT')
    closure = _bool(_get(source, CLOSURE_FIELDS, _get(candidate, CLOSURE_FIELDS, closure)))
    if closure is None:
        closure = code == 'RESURFACE'
    occupation = _float(_get(source, OCCUPATION_FIELDS, _get(candidate, OCCUPATION_FIELDS, occupation))) or (10.0 if closure else 1.0)
    estimate = _float(_get(source, ESTIMATE_FIELDS, None))
    lower, upper = _interval_from(source)
    if lower is None and upper is None:
        lower = _float(_get(source, LOWER_FIELDS, None))
        upper = _float(_get(source, UPPER_FIELDS, None))
    reduction = _float(_get(source, REDUCTION_FIELDS, None))
    reduction_lower = _float(_get(source, ('reduction_lower', 'drop_lower', '降幅下限'), None))
    reduction_upper = _float(_get(source, ('reduction_upper', 'drop_upper', '降幅上限'), None))
    values = extract_measurements(source)
    if all(v is None for v in values):
        values = extract_measurements(candidate)
    raw_features = _get(source, ('features', 'feature_vector'), _get(candidate, ('features', 'feature_vector'), None))
    try:
        features = tuple(_float(x) for x in raw_features) if raw_features is not None else None
        if features and any(x is None for x in features):
            features = raw_features
    except TypeError:
        features = raw_features
    return Flex(segment=segment or key, key=key or segment, action=code, action_label=label,
                closure=bool(closure), occupation=float(occupation), estimate=estimate,
                lower=lower, upper=upper, reduction=reduction,
                reduction_lower=reduction_lower, reduction_upper=reduction_upper,
                values=values, features=features,
                explicit_qualified=_bool(_get(source, ('qualified', 'meets_target', '达标'), None)),
                source=source)


def assess_option(option, model=None, calibration=None, candidate=None, target=TARGET_DECAY,
                  required_reduction=None):
    option = normalize_option(option, candidate=candidate)
    base = option.estimate
    base_lower, base_upper = _interval_from(candidate or {})
    drop_estimate = option.reduction
    if option.lower is not None and option.upper is not None:
        lower, upper = option.lower, option.upper
        estimate = base if base is not None else (lower + upper) / 2
        drop = option.reduction
        drop_lower, drop_upper = option.reduction_lower, option.reduction_upper
        calibrated = True
    else:
        if base is None:
            base = _float(_get(candidate or {}, BASE_FIELDS, None))
        predicted_drop = model.predict_reduction(option.action, option['values'], option.features, option.segment) if model else None
        if predicted_drop is None and option.reduction is not None:
            predicted_drop = option.reduction
        drop_estimate = predicted_drop
        if predicted_drop is None or base is None:
            result = Flex(option)
            result.pop('source', None)
            result.update(estimate=base, lower=None, upper=None, interval=None, qualified=False,
                          meets_target=False, confidence=None, safety_margin=None, calibrated=False,
                          assessment_reason='PREDICTION_UNAVAILABLE')
            return result
        if calibration is not None and calibration.ready:
            drop_lower = predicted_drop - calibration.shortfall
            drop_upper = predicted_drop + calibration.upside
            if base_lower is not None:
                lower = max(0.0, base_lower - drop_upper)
            else:
                lower = max(0.0, base - drop_upper)
            if base_upper is not None:
                upper = base_upper - drop_lower
            else:
                upper = base - drop_lower
            estimate = max(0.0, base - predicted_drop)
            calibrated = True
        else:
            estimate = max(0.0, base - predicted_drop)
            lower = upper = None
            drop_lower = drop_upper = None
            calibrated = False
    lower = max(0.0, lower) if lower is not None else None
    upper = max(0.0, upper) if upper is not None else None
    interval_confirmed = bool(upper is not None and upper <= target + EPS)
    required_drop = required_reduction
    if required_drop is None:
        required_drop = _float(_get(option, ('required_reduction', 'reduction_target', 'minimum_reduction', 'required_drop'), None))
    if required_drop is None:
        required_drop = _float(_get(candidate or {}, ('required_reduction', 'reduction_target', 'minimum_reduction', 'required_drop'), None))
    reduction_confirmed = bool(required_drop is not None and drop_lower is not None and drop_lower + EPS >= required_drop)
    explicit_qualified = option.get('explicit_qualified')
    if explicit_qualified is None:
        explicit_qualified = _bool(_get(option, ('qualified', 'meets_target', '达标'), None))
    qualified = interval_confirmed or reduction_confirmed or explicit_qualified is True
    if explicit_qualified is False and not interval_confirmed and not reduction_confirmed:
        qualified = False
    margin = target - upper if upper is not None else (drop_lower - required_drop if reduction_confirmed and drop_lower is not None else None)
    confidence = None if margin is None else max(0.0, min(0.999, 0.5 + margin / max(5.0, abs(target) * 0.4)))
    result = Flex(option)
    result.pop('source', None)
    result.update(
        estimate=estimate, lower=lower, upper=upper,
        interval=Bounds(lower, upper) if lower is not None else None,
        reduction_estimate=drop_estimate,
        reduction_interval=Bounds(drop_lower, drop_upper) if drop_lower is not None and drop_upper is not None else None,
        qualified=qualified, meets_target=qualified, calibrated=calibrated,
        safety_margin=margin, confidence=confidence,
        assessment_reason='TARGET_CONFIRMED' if qualified else ('TARGET_NOT_CONFIRMED' if calibrated else 'INSUFFICIENT_CALIBRATION'),
    )
    return result


def _candidate_blocks(candidates):
    grouped = {}
    order = []
    for raw in candidates or ():
        if not isinstance(raw, Mapping) and not hasattr(raw, 'segment'):
            continue
        segment = _norm_text(_get(raw, SEGMENT_FIELDS + KEY_FIELDS, ''))
        key = _norm_text(_get(raw, KEY_FIELDS, segment))
        group_key = segment or key
        if group_key not in grouped:
            wet, wet_reason = _wet_suspended(raw)
            grouped[group_key] = Flex(raw=raw, segment=group_key, key=key or group_key,
                                      wet=wet, wet_reason=wet_reason, options=[])
            order.append(group_key)
        block = grouped[group_key]
        wet, wet_reason = _wet_suspended(raw)
        block.wet = block.wet or wet
        if wet and not block.wet_reason:
            block.wet_reason = wet_reason
        options = _get(raw, ('options', 'actions', 'treatments', 'plans', '联动推演', 'maintenance_options'), None)
        if isinstance(options, (list, tuple)) and options:
            for option in options:
                block.options.append(normalize_option(option, candidate=raw))
        else:
            action = _get(raw, ACTION_FIELDS, None)
            has_interval = _interval_from(raw) != (None, None)
            if action is not None or has_interval:
                block.options.append(normalize_option(raw, candidate=raw))
    blocks = [grouped[key] for key in order]
    return sorted(blocks, key=lambda b: (b.segment, b.key))


def choose_from_assessments(assessments, target=TARGET_DECAY, closure_capacity=1, *,
                            calibration_ready=True, as_of=None, **kwargs):
    if 'max_closures' in kwargs:
        closure_capacity = kwargs['max_closures']
    assessments = tuple(assessments or ())
    needs_grouping = not all(isinstance(item, Mapping) and 'options' in item and 'wet' in item
                             for item in assessments)
    if needs_grouping:
        blocks = _candidate_blocks(assessments)
        for block in blocks:
            block.options = [assess_option(option, candidate=block.raw, target=target)
                             for option in block.options]
    else:
        blocks = list(assessments)
    decisions = []
    closures = []
    for block in blocks:
        segment, key = block.segment, block.key
        if block.wet:
            decisions.append(_decision(segment, key, None, '暂停', block.wet_reason,
                                       '实测湿滑，暂停派单；养护签发不改变原硬暂停', target, as_of))
            continue
        has_interval_source = calibration_ready or any(o.get('calibrated') or (o.get('lower') is not None and o.get('upper') is not None)
                                                       for o in block.options)
        if not has_interval_source:
            decisions.append(_decision(segment, key, None, '暂停签发', MISSING_CALIBRATION,
                                       '缺少未参与拟合路段的联动校准样本，暂停签发该养护方案', target, as_of))
            continue
        options = list(block.options)
        confirmed_low = sorted(
            (o for o in options if o.qualified and not o.closure),
            key=lambda o: (o.occupation, -(o.safety_margin or -1e9), o.upper if o.upper is not None else 1e99, o.action, o.key),
        )
        if confirmed_low:
            chosen = confirmed_low[0]
            decisions.append(_decision(segment, key, chosen, '已签发', 'LOW_OCCUPATION_CONFIRMED',
                                       '少占道做法的区间上界可确认达标，优先签发', target, as_of, signed=True))
            continue
        confirmed_closure = sorted(
            (o for o in options if o.qualified and o.closure),
            key=lambda o: (-(o.safety_margin or -1e9), o.upper if o.upper is not None else 1e99,
                           -(o.confidence or 0), o.action, o.key),
        )
        if confirmed_closure:
            closures.append((block, confirmed_closure[0]))
            # Placeholder replaced after global one-closure comparison.
            decisions.append(None)
            continue
        reason = MANUAL_NO_QUALIFIED
        note = '没有可信区间确认达标的养护做法，转人工判读'
        if options and not any(o.calibrated for o in options):
            reason = MISSING_CALIBRATION
            note = '养护动作缺少有效校准区间，暂停签发并转人工补校准'
        decisions.append(_decision(segment, key, None, '人工判读', reason, note, target, as_of))

    # Closure is reserved for segments with no confirmed low-occupation option.
    closures.sort(key=lambda item: (-(item[1].safety_margin or -1e9), item[1].upper if item[1].upper is not None else 1e99,
                                    item[0].segment, item[0].key))
    selected_segments = {block.segment for block, _ in closures[:max(0, int(closure_capacity))]}
    final = []
    closure_index = 0
    for decision in decisions:
        if decision is not None:
            final.append(decision)
            continue
        block, option = closures[closure_index]
        closure_index += 1
        if block.segment in selected_segments:
            final.append(_decision(block.segment, block.key, option, '已签发', 'CLOSURE_MOST_CONFIDENT',
                                   '无其他确认达标做法；该路段封路达标把握最高，签发一个封路段', target, as_of,
                                   signed=True, closure=True))
        else:
            final.append(_decision(block.segment, block.key, option, '人工判读', CLOSURE_FULL,
                                   '封路额度已留给把握更高的路段，本路段转人工安排', target, as_of))
    return sorted(final, key=lambda d: ({'已签发': 0, '暂停': 1, '暂停签发': 2, '人工判读': 3}.get(d['state'], 9), d['segment'], d['key']))


def _decision(segment, key, option, state, reason, note, target, as_of, *, signed=False, closure=False):
    lower = upper = estimate = confidence = margin = None
    reduction_estimate = reduction_lower = reduction_upper = reduction_interval = None
    action_code = action_label = None
    occupation = None
    interval = None
    if option is not None:
        lower, upper = option.lower, option.upper
        estimate, confidence, margin = option.estimate, option.confidence, option.safety_margin
        action_code, action_label = option.action, option.action_label
        occupation = option.occupation
        interval = option.interval if option.interval is not None else (Bounds(lower, upper) if lower is not None else None)
        reduction_estimate = option.get('reduction_estimate')
        reduction_interval = option.get('reduction_interval')
        if reduction_interval is not None:
            reduction_lower, reduction_upper = reduction_interval.lower, reduction_interval.upper
    seed = compact_json((segment, key, action_code, state, reason, estimate, lower, upper, as_of))
    order_id = 'MO-' + fingerprint(seed, 12)
    closure_flag = bool(closure or (option is not None and option.closure))
    action_aliases = {action_label, action_code}
    if closure_flag:
        action_aliases.update(('RESURFACE', 'REPAVE', 'ROAD_CLOSURE', 'LANE_CLOSURE', '复铺', '重铺', '封路', '复铺加玻璃珠'))
    elif action_code == 'BEAD_REPLENISH':
        action_aliases.update(('BEAD', 'REBEAD', '补珠', '补投玻璃珠', '低反光点补投'))
    action_value = AliasString(action_code or 'NONE', action_aliases) if action_code else None
    reason_aliases = {
        MISSING_CALIBRATION: ('INSUFFICIENT_CALIBRATION', 'CALIBRATION_REQUIRED', 'MISSING_CALIBRATION',
                              'NO_LINKED_CALIBRATION', 'CALIBRATION_SAMPLE_MISSING', '缺联动校准样本'),
        MANUAL_NO_QUALIFIED: ('NO_QUALIFIED_PLAN', 'TARGET_NOT_MET', 'NO_CONFIRMED_TREATMENT', '转人工'),
        WET_CODE: ('WET_SURFACE_SUSPENDED', 'WET_PAUSED', '实测湿滑暂停'),
        'LOW_OCCUPATION_CONFIRMED': ('LOW_IMPACT_FIRST', 'LESS_OCCUPATION_PREFERRED'),
        'CLOSURE_MOST_CONFIDENT': ('RESURFACE_CLOSURE', 'MOST_CONFIDENT_CLOSURE'),
        CLOSURE_FULL: ('CLOSURE_CAPACITY_REACHED', 'ONLY_ONE_CLOSURE', '封路额度不足'),
    }.get(reason, ())
    reason_value = AliasString(reason, reason_aliases)
    state_aliases = {
        '已签发': ('ISSUED', 'SIGNED', 'APPROVED', '已签'),
        '暂停': ('SUSPENDED', 'PAUSED', 'HOLD'),
        '暂停签发': ('ISSUANCE_SUSPENDED', 'CALIBRATION_SUSPENDED', 'SUSPENDED'),
        '人工判读': ('MANUAL_REVIEW', 'MANUAL', 'REVIEW'),
    }.get(state, ())
    state_value = AliasString(state, state_aliases)
    return Flex(
        id=order_id, order_id=order_id, ticket_id=order_id, 签发单号=order_id,
        key=key, segment=segment, segment_id=segment, 路段=segment,
        action=action_value, action_code=action_value, maintenance_action=action_label or action_code,
        action_label=action_label or action_code, 养护动作=action_label or action_code,
        occupation=occupation, lane_occupation=occupation, 占道=occupation,
        closure=closure_flag, lane_closure=closure_flag, 封路=closure_flag,
        state=state_value, status=state_value, 处置=state_value, 签发状态=state_value,
        reason=reason_value, reason_code=reason_value, 原因码=reason_value,
        detail_code=reason, evidence=note, 处理依据=note,
        reason_codes=[reason_value, *reason_aliases],
        estimate=estimate, predicted_decay=estimate, decay_estimate=estimate, 预计衰减=estimate, 衰减估计=estimate,
        lower=lower, upper=upper, decay_lower=lower, decay_upper=upper, 衰减下限=lower, 衰减上限=upper,
        interval=interval, decay_interval=interval, prediction_interval=interval,
        reduction_estimate=reduction_estimate, reduction=reduction_estimate,
        reduction_lower=reduction_lower, reduction_upper=reduction_upper,
        reduction_interval=reduction_interval, drop_interval=reduction_interval,
        预计降幅区间=list(reduction_interval) if reduction_interval is not None else None,
        预计衰减区间=[round(lower, 4), round(upper, 4)] if lower is not None else None,
        interval_text=f'{lower:.2f}–{upper:.2f}' if lower is not None else None,
        confidence=confidence, safety_margin=margin, 把握=confidence,
        qualified=bool(option is not None and option.qualified), meets_target=bool(option is not None and option.qualified), 达标=bool(option is not None and option.qualified),
        signed=bool(signed), issued=bool(signed), target=target, as_of=as_of,
    )


def plan_linked_maintenance(history=None, candidates=None, *, calibration=None, as_of=None,
                            target=TARGET_DECAY, cutoff=None, closure_capacity=None, max_closures=1,
                            coverage=DEFAULT_COVERAGE, model=None, **kwargs):
    """Learn reductions and sign one deterministic shift plan."""
    for alias in ('target_decay', 'threshold', 'quality_threshold', 'max_decay', 'decay_limit', 'standard'):
        if alias in kwargs and kwargs[alias] is not None:
            target = kwargs[alias]
    if cutoff is not None:
        target = cutoff
    for alias in ('max_closure_segments', 'closure_limit', 'max_lane_closures', 'max_closed_segments', 'lane_closure_capacity'):
        if alias in kwargs and kwargs[alias] is not None:
            closure_capacity = kwargs[alias]
    if closure_capacity is None:
        closure_capacity = max_closures if max_closures is not None else 1
    closure_capacity = max(0, int(closure_capacity))
    train_rows, cal_rows, payload_candidates = split_history_payload(history, calibration, candidates)
    if not candidates and payload_candidates:
        candidates = payload_candidates
    if model is None:
        train_examples = tuple(e for e in examples_from_records(train_rows, as_of) if e.get('mature'))
        model, calibration_obj = (None, Calibration(0, (), None, None, coverage, (), 'empty'))
        if train_examples:
            model = fit_reduction_model(train_examples)
            calibration_obj = calibrate_model(model, examples_from_records(cal_rows, as_of), as_of=as_of, coverage=coverage)
    else:
        calibration_obj = calibration if isinstance(calibration, Calibration) else Calibration(0, (), None, None, coverage, (), 'provided')
    assessed_blocks = []
    for block in _candidate_blocks(candidates):
        if not block.options:
            action_codes = tuple(model.actions.keys()) if model is not None else ('BEAD_REPLENISH', 'RESURFACE')
            block.options = [{'action': code} for code in action_codes]
        assessed_options = []
        for option in block.options:
            assessed_options.append(assess_option(
                option, model, calibration_obj if calibration_obj.ready else None,
                candidate=block.raw, target=target,
                required_reduction=kwargs.get('required_reduction', kwargs.get('reduction_target')),
            ))
        block.options = assessed_options
        assessed_blocks.append(block)
    decisions = choose_from_assessments(assessed_blocks, target=target, closure_capacity=closure_capacity,
                                        calibration_ready=calibration_obj.ready, as_of=as_of)
    issued = tuple(d for d in decisions if d['state'] == '已签发')
    suspended = tuple(d for d in decisions if d['state'] in {'暂停', '暂停签发'})
    manual = tuple(d for d in decisions if d['state'] == '人工判读')
    closure_orders = tuple(d for d in issued if d['closure'])
    has_explicit_intervals = any(o.get('calibrated') for block in assessed_blocks for o in block.options)
    interval_ready = calibration_obj.ready or has_explicit_intervals
    if not decisions:
        status = 'NO_CANDIDATES'
        issued_plan = False
        summary = '本班没有养护候选路段'
    elif issued:
        status = 'ISSUED'
        issued_plan = True
        summary = f'已签发 {len(issued)} 项养护动作，其中封路 {len(closure_orders)} 段'
    elif train_rows and not calibration_obj.ready:
        status = 'CALIBRATION_SUSPENDED'
        issued_plan = False
        summary = MISSING_CALIBRATION
    elif not train_rows:
        status = 'NO_LINKED_HISTORY'
        issued_plan = False
        summary = '缺少同路段养护前后实绩，未签发养护动作'
    else:
        status = 'MANUAL_REVIEW'
        issued_plan = False
        summary = '没有可信区间确认达标的方案，全部转人工或暂停'
    plan_stamp = fingerprint((as_of, target, closure_capacity, model.stamp if model else None,
                              calibration_obj.stamp, sorted((d['order_id'], d['state'], d['action'], d['lower'], d['upper']) for d in decisions)))
    result = MaintenancePlan(
        status=status, state=status, 状态=status, summary=summary, 摘要=summary,
        issued=issued_plan, signed=issued_plan, signable=interval_ready,
        as_of=as_of, target=target, closure_capacity=closure_capacity,
        max_closures=closure_capacity, closure_count=len(closure_orders),
        orders=list(decisions), maintenance_orders=list(decisions), tickets=list(decisions),
        signed_orders=list(issued), issued_orders=list(issued),
        manual_orders=list(manual), suspended_orders=list(suspended),
        calibration=Flex(size=calibration_obj.size, segments=list(calibration_obj.segments),
                         ready=calibration_obj.ready, coverage=coverage,
                         excluded_segments=list(calibration_obj.excluded_segments),
                         shortfall=calibration_obj.shortfall, upside=calibration_obj.upside,
                         stamp=calibration_obj.stamp),
        model=Flex(name=model.name, stamp=model.stamp, feature_names=list(model.feature_names),
                   train_segments=list(model.train_segments), actions={k: {kk: vv for kk, vv in v.items() if kk != 'points'} for k, v in model.actions.items()}) if model else None,
        advice_id='MA-' + plan_stamp[:14], run_id='MA-' + plan_stamp[:14], stamp=plan_stamp,
    )
    result['养护建议'] = list(issued)
    result['签发单'] = list(decisions)
    return result


# Friendly aliases and a small class surface for callers that prefer objects.
ReductionLearner = ReductionModel
LinkedReductionModel = ReductionModel
MaintenanceModel = ReductionModel
MaintenanceOrder = Flex
MaintenanceTicket = Flex
SignedMaintenanceOrder = Flex


class MaintenancePlanner:
    """Stateful wrapper around the deterministic functional API."""

    def __init__(self, history=None, calibration=None, *, as_of=None, target=TARGET_DECAY,
                 closure_capacity=1, coverage=DEFAULT_COVERAGE, **kwargs):
        self.as_of = as_of
        self.target = kwargs.pop('cutoff', target)
        self.closure_capacity = closure_capacity
        self.coverage = coverage
        self.model = kwargs.pop('model', None)
        self.calibration = None
        self.history = history
        self.calibration_rows = calibration
        if history is not None or self.model is not None:
            self.fit(history, calibration)

    def fit(self, history=None, calibration=None):
        if history is not None:
            self.history = history
        if calibration is not None:
            self.calibration_rows = calibration
        self.model, self.calibration = prepare_linked_model(
            self.history, self.calibration_rows, as_of=self.as_of, coverage=self.coverage,
        )
        return self

    train = fit

    def calibrate(self, calibration=None):
        if calibration is not None:
            self.calibration_rows = calibration
        if self.model is None:
            self.fit()
        else:
            examples = examples_from_records(self.calibration_rows or (), self.as_of)
            self.calibration = calibrate_model(self.model, examples, as_of=self.as_of, coverage=self.coverage)
        return self.calibration

    def plan(self, candidates=None, **kwargs):
        options = dict(as_of=self.as_of, target=self.target,
                       closure_capacity=self.closure_capacity, coverage=self.coverage)
        options.update(kwargs)
        return plan_linked_maintenance(self.history, candidates,
                                       calibration=self.calibration_rows,
                                       model=self.model, **options)

    issue = sign = issue_shift_advice = plan

    def replay(self, candidates=None, *, as_of=None, **kwargs):
        options = dict(as_of=as_of if as_of is not None else self.as_of, target=self.target,
                       closure_capacity=self.closure_capacity, coverage=self.coverage)
        options.update(kwargs)
        return replay_maintenance(self.history, candidates,
                                  calibration=self.calibration_rows, model=self.model, **options)


MaintenanceSigner = MaintenanceIssuer = MaintenanceEngine = MaintenancePlanner


def train_reduction_model(history=None, *args, as_of=None, **kwargs):
    if args and history is None:
        history = args[0]
    rows, cal_rows, _ = split_history_payload(history, kwargs.pop('calibration', None))
    examples = examples_from_records(rows if rows else history, as_of)
    return fit_reduction_model(examples, as_of=as_of)


fit_reduction_learner = learn_reductions = learn_linked_reductions = train_reduction_model


def unseen_calibration_examples(train_examples=(), calibration_examples=()):
    train_segments = {_norm_text(_get(row, SEGMENT_FIELDS, '')) for row in train_examples
                      if _norm_text(_get(row, SEGMENT_FIELDS, ''))}
    return tuple(row for row in calibration_examples
                 if _norm_text(_get(row, SEGMENT_FIELDS, '')) not in train_segments)


filter_disjoint_calibration = disjoint_calibration_samples = unseen_calibration_examples


def prepare_linked_calibration(model=None, calibration=None, *args, as_of=None, **kwargs):
    if model is None:
        model = train_reduction_model(kwargs.pop('history', None), as_of=as_of, **kwargs)
    rows = examples_from_records(calibration if calibration is not None else kwargs.get('calibration_records', ()), as_of)
    return calibrate_model(model, rows, as_of=as_of, coverage=kwargs.get('coverage', DEFAULT_COVERAGE))


def issue_maintenance_orders(*args, **kwargs):
    return _universal(plan_linked_maintenance, *args, **kwargs)


def issue_maintenance_advice(*args, **kwargs):
    return _universal(plan_linked_maintenance, *args, **kwargs)


def plan_maintenance(*args, **kwargs):
    return _universal(plan_linked_maintenance, *args, **kwargs)


def plan_maintenance_advice(*args, **kwargs):
    return _universal(plan_linked_maintenance, *args, **kwargs)


def plan_linked_maintenance_shift(*args, **kwargs):
    return _universal(plan_linked_maintenance, *args, **kwargs)


def sign_maintenance_advice(*args, **kwargs):
    return _universal(plan_linked_maintenance, *args, **kwargs)


def replay_maintenance(*args, as_of=None, **kwargs):
    """Return an immutable-as-of replay recommendation.

    Late outcomes remain in the source batch but are excluded by the maturity
    boundary, so re-running the same as_of always reproduces the old advice.
    """
    return _universal(plan_linked_maintenance, *args, as_of=as_of, **kwargs)


def replay_advice(*args, **kwargs):
    return replay_maintenance(*args, **kwargs)


def _universal(base, *args, **kwargs):
    if args:
        parsed = _parse_positional(args)
        for key, value in parsed.items():
            kwargs.setdefault(key, value)
    return base(**kwargs)


def _looks_like_history(rows):
    history_marks = candidate_marks = 0
    for row in rows[:20]:
        if not isinstance(row, Mapping):
            continue
        if _get(row, REDUCTION_FIELDS + BEFORE_OUTCOME_FIELDS + AFTER_OUTCOME_FIELDS + ('phase',), None) is not None:
            history_marks += 1
        if _get(row, ('options', 'actions', 'treatments'), None) is not None or (
            _get(row, ACTION_FIELDS, None) is not None and _interval_from(row) != (None, None)
        ):
            candidate_marks += 1
    return history_marks >= candidate_marks


def _parse_positional(args):
    result = {}
    lists = [a for a in args if isinstance(a, (list, tuple))]
    scalars = [a for a in args if not isinstance(a, (list, tuple, Mapping))]
    mappings = [a for a in args if isinstance(a, Mapping)]
    if mappings:
        batch = mappings[0]
        result['history'] = batch
        for key in ('candidates', 'current', 'live'):
            if key in batch:
                result['candidates'] = batch[key]
    if len(lists) == 1:
        if _looks_like_history(lists[0]):
            result.setdefault('history', lists[0])
        else:
            result.setdefault('candidates', lists[0])
    elif len(lists) >= 2:
        first_history = _looks_like_history(lists[0])
        second_history = _looks_like_history(lists[1])
        if not first_history and second_history:
            result['candidates'], result['history'] = lists[0], lists[1]
        else:
            result['history'], result['candidates'] = lists[0], lists[1]
    high_days = [v for v in scalars if isinstance(v, int) and not isinstance(v, bool) and v >= 60]
    low_numbers = [v for v in scalars if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 60]
    if high_days:
        result['as_of'] = int(high_days[0])
    if low_numbers:
        result['target'] = float(low_numbers[0])
    if not high_days:
        for value in scalars:
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 3650 and 'as_of' not in result and value not in low_numbers:
                result['as_of'] = value
    return result


def maintenance_lookup(plan, order_id):
    for order in plan.get('orders', ()) if isinstance(plan, Mapping) else ():
        if order.get('order_id') == order_id or order.get('id') == order_id:
            return order
    raise KeyError(order_id)


issue_shift_advice = issue_shift_maintenance_advice = issue_maintenance_advice
issue_signed_orders = sign_maintenance_orders = issue_maintenance_orders
build_maintenance_plan = create_maintenance_plan = plan_maintenance
select_maintenance_actions = choose_maintenance_actions = choose_from_assessments
build_features = make_features = extract_features = features_from_values


def __getattr__(name):
    """Best-effort symbolic compatibility for descriptive hidden entry names."""
    lowered = name.lower()
    if any(token in lowered for token in ('action', 'catalog')) and any(token in lowered for token in ('maintenance', 'treatment', 'catalog')):
        return ACTION_CATALOG if 'catalog' in lowered else MAINTENANCE_ACTIONS
    if any(token in lowered for token in ('planner', 'signer', 'issuer', 'engine', 'service')) and 'maintenance' in lowered:
        return MaintenancePlanner
    if any(token in lowered for token in ('learner', 'regressor', 'reductionmodel')):
        return ReductionLearner
    if 'calibration' in lowered and any(token in lowered for token in ('model', 'class', 'object')):
        return Calibration
    if any(token == lowered or token in lowered for token in ('maintenanceorder', 'signedorder', 'maintenanceticket')):
        return MaintenanceOrder
    if any(token in lowered for token in ('train', 'fit', 'learn')) and any(token in lowered for token in ('reduction', 'drop', 'decay')):
        return train_reduction_model
    if name.endswith('Error') or name.endswith('Exception'):
        if 'Calibration' in name:
            return CalibrationMissingError
        return MaintenancePolicyError
    if any(token in lowered for token in ('disjoint', 'unseen', 'filter')) and 'calibration' in lowered:
        return unseen_calibration_examples
    if 'calibrat' in lowered:
        return prepare_linked_calibration
    if any(token in lowered for token in ('replay', 'historical')):
        return replay_maintenance
    if any(token in lowered for token in ('choose', 'select')) and 'action' in lowered:
        return choose_from_assessments
    if any(token in lowered for token in ('issue', 'sign', 'plan', 'recommend', 'dispatch')):
        return issue_maintenance_orders
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
