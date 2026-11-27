from argparse import ArgumentParser
from functools import lru_cache
import json
from pathlib import Path
import re

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge, UnsupportedMediaType

from .pipeline import DISPLAY, MODULES, execute, open_fixture, preview

app = Flask(__name__, static_folder='web/static', template_folder='web/templates')
app.config['MAX_CONTENT_LENGTH'] = 1_500_000
app.json.sort_keys = False
MAINTENANCE_STORE = {}


def validate_parameters(capacity, strategy):
    if type(capacity) is not int or not 1 <= capacity <= 60:
        raise ValueError('额度应是 1 到 60 的整数')
    if not isinstance(strategy, str) or strategy not in {'balanced', 'focus'}:
        raise ValueError('策略应是 balanced 或 focus')
    return capacity, strategy


def remember_maintenance(result):
    plan = result.get('maintenance_advice', result) if isinstance(result, dict) else None
    if not isinstance(plan, dict):
        return
    if plan.get('advice_id'):
        MAINTENANCE_STORE[plan['advice_id']] = plan
    for order in plan.get('orders', ()):  # all signed/manual/suspended decisions are queryable
        if order.get('order_id'):
            MAINTENANCE_STORE[order['order_id']] = order


@lru_cache(maxsize=8)
def sample(capacity=12, strategy='balanced'):
    return preview(capacity, strategy)


@app.get('/')
def home():
    return render_template('desk.html', modules=MODULES, display=DISPLAY)


@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'linebeads'}


@app.get('/api/linebeads/overview')
def overview():
    return jsonify(sample())


@app.post('/api/linebeads/run')
def run_case():
    try:
        posted = request.get_json()
        if not isinstance(posted, dict):
            raise ValueError('请求对象无效')
        # Unknown keys are tolerated because linked-maintenance callers may
        # supply their own history/candidate schema; execute() ignores metadata
        # it does not understand.
        mode = posted.get('strategy', 'balanced')
        capacity, strategy = validate_parameters(posted.get('capacity', 12), mode)
        standard_keys = {'capacity', 'strategy'}
        extra_keys = set(posted) - standard_keys
        if 'records' not in posted and extra_keys <= {'as_of'}:
            if 'as_of' in posted:
                raise ValueError('导入截止日需要同时提交 records')
            return jsonify(sample(capacity, strategy))
        as_of = posted.get('as_of', 96)
        if type(as_of) is not int or not 91 <= as_of <= 3650:
            raise ValueError('截止日须在第 91 日后')
        if 'records' not in posted:
            from .maintenance_dispatch import plan_linked_maintenance
            report = plan_linked_maintenance(
                posted, as_of=as_of,
                target=posted.get('target_decay', posted.get('target', posted.get('threshold', 25))),
                closure_capacity=posted.get('closure_capacity', posted.get('max_closures', 1)),
            )
            remember_maintenance(report)
            return jsonify(report)
        result = execute(posted['records'], capacity, strategy, as_of=as_of,
                         maintenance_payload=posted)
        remember_maintenance(result)
        return jsonify(result)
    except (BadRequest, UnsupportedMediaType):
        return jsonify(code='INVALID_JSON', message='请提交 JSON 对象'), 400
    except ValueError as issue:
        return jsonify(code='INVALID_BATCH', message=str(issue)), 422


def plan_posted_maintenance(posted):
    from .maintenance_dispatch import plan_linked_maintenance
    as_of = posted.get('as_of', 96) if isinstance(posted, dict) else 96
    if type(as_of) is not int or not 0 <= as_of <= 3650:
        raise ValueError('截止日须为 0 至 3650 的整数工日')
    report = plan_linked_maintenance(
        posted,
        as_of=as_of,
        target=posted.get('target_decay', posted.get('target', posted.get('threshold', 25))),
        closure_capacity=posted.get('closure_capacity', posted.get('max_closures', 1)),
    )
    remember_maintenance(report)
    return report


@app.post('/api/linebeads/maintenance')
@app.post('/api/linebeads/maintenance/issue')
@app.post('/api/linebeads/maintenance/plan')
@app.post('/api/linebeads/linked-maintenance/issue')
@app.post('/api/maintenance/issue')
@app.post('/api/maintenance/plan')
def maintenance_run():
    try:
        posted = request.get_json()
        if not isinstance(posted, dict):
            raise ValueError('请求对象无效')
        return jsonify(plan_posted_maintenance(posted))
    except (BadRequest, UnsupportedMediaType):
        return jsonify(code='INVALID_JSON', message='请提交 JSON 对象'), 400
    except ValueError as issue:
        return jsonify(code='INVALID_BATCH', message=str(issue)), 422


def maintenance_ticket(order_id):
    if order_id not in MAINTENANCE_STORE:
        remember_maintenance(sample())
    ticket = MAINTENANCE_STORE.get(order_id)
    if ticket is None:
        return jsonify(code='UNKNOWN_MAINTENANCE_ORDER', message='未找到养护签发单'), 404
    return jsonify(ticket)


@app.get('/api/linebeads/maintenance/<order_id>')
def maintenance_detail(order_id):
    return maintenance_ticket(order_id)


@app.get('/api/linebeads/maintenance-orders/<order_id>')
def maintenance_order_detail(order_id):
    return maintenance_ticket(order_id)


@app.get('/api/linebeads/orders/<order_id>')
@app.get('/api/maintenance-orders/<order_id>')
@app.get('/api/maintenance/orders/<order_id>')
def generic_order_detail(order_id):
    return maintenance_ticket(order_id)


@app.get('/api/linebeads/stages/<stage>')
def stage_detail(stage):
    mapping = {slug: index for index, (slug, _) in enumerate(MODULES)}
    if stage not in mapping:
        return jsonify(code='UNKNOWN_STAGE', message='未找到业务环节'), 404
    try:
        raw = request.args.get('capacity', '12')
        if not re.fullmatch(r'[0-9]{1,2}', raw):
            raise ValueError('额度应是 1 到 60 的整数')
        capacity, strategy = validate_parameters(int(raw), request.args.get('strategy', 'balanced'))
    except ValueError as issue:
        return jsonify(code='INVALID_BATCH', message=str(issue)), 422
    result = sample(capacity, strategy)
    return jsonify(run_id=result['run_id'], **result['sections'][mapping[stage]])


@app.errorhandler(RequestEntityTooLarge)
def rejected_size(_issue):
    return jsonify(code='PAYLOAD_TOO_LARGE', message='单据批次超过 1.5 MB'), 413


def main():
    parser = ArgumentParser(description=DISPLAY['title'])
    commands = parser.add_subparsers(dest='command', required=True)
    runner = commands.add_parser('evaluate', help='复算一个班次，并输出处置单与审计证据')
    runner.add_argument('--capacity', type=int, default=12)
    runner.add_argument('--strategy', choices=('balanced', 'focus'), default='balanced')
    runner.add_argument('--input', type=Path)
    runner.add_argument('--output', type=Path)
    server = commands.add_parser('serve', help='打开本地工作台')
    server.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.command == 'serve':
        from waitress import serve
        sample()
        serve(app, host='0.0.0.0', port=args.port, threads=4)
        return
    try:
        capacity, strategy = validate_parameters(args.capacity, args.strategy)
        if args.input:
            custom = open_fixture(args.input)
            result = execute(custom['records'], capacity, strategy, as_of=custom.get('as_of', 96))
        else:
            result = sample(capacity, strategy)
    except (ValueError, OSError) as issue:
        parser.error(str(issue))
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
