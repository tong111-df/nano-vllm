"""Compare Prefill budgets and yield limits using the unified CPU runner."""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

from experiments.run import ROOT, resolve_config, run_experiment


DEFAULT_CONFIG = ROOT / 'experiments' / 'configs' / 'unit8_arrivals.json'


def run_sweep(raw, budgets=(1, 2, 5), output_root=None):
    config = resolve_config(raw)
    if config['workload'] != 'arrivals':
        raise ValueError('Unit 8 sweep requires the arrivals workload')
    if not budgets or any(type(b) is not int or b < 1 for b in budgets):
        raise ValueError('budgets must contain positive integers')
    parent = Path(output_root) if output_root is not None else ROOT / 'results' / 'unit8'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = parent / (stamp + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for budget in budgets:
        current = deepcopy(config)
        current['parameters']['budget'] = budget
        directory, report = run_experiment(current, output)
        for index, case in enumerate(report['cases']):
            rows.append({
                'budget': budget, **case['strategy'], **case['result']['summary'],
                'report': (directory / 'report.json').relative_to(output).as_posix(),
                'case_index': index,
            })
    # Written last: if a case fails, partial raw reports remain, without a
    # summary that could be mistaken for a completed sweep.
    (output / 'summary.json').write_text(json.dumps({
        'units': 'Scheduling rounds with synthetic output; not GPU latency.',
        'base_config': config, 'budgets': list(budgets), 'rows': rows,
    }, indent=2) + '\n', encoding='utf-8')
    with (output / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 2, 5])
    args = parser.parse_args()
    try:
        output, rows = run_sweep(json.loads(args.config.read_text(encoding='utf-8-sig')),
                                 args.budgets)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'Sweep failed: {error}\n')
    print('budget policy          limit A_first_decode A_finish B_max_wait all_finished')
    for row in rows:
        limit = str(row['max_consecutive_prefills']) if row['policy'] == 'bounded_prefill' else '-'
        print(f"{row['budget']:>6} {row['policy']:<15} {limit:>5} "
              f"{row['A_first_decode_round']:>14} {row['A_finish_round']:>8} "
              f"{row['B_max_first_token_wait']:>10} {row['all_finished_round']:>12}")
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
