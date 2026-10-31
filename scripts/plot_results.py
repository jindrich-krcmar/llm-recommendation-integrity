"""Render the five-condition comparison from exported figure data."""
import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, default=ROOT / 'results/published/plotted_metrics.csv')
    parser.add_argument('--output-directory', type=Path, default=ROOT / 'outputs/figures')
    args = parser.parse_args()
    with args.metrics.open(encoding='utf-8', newline='') as handle:
        data = list(csv.DictReader(handle))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = ['M0', 'task_aligned_sft', 'biased_sft', 'biased_sft_dpo', 'biased_sft_dpo_sft']
    labels = ['Original', 'Aligned\nSFT', 'Biased\nSFT', 'Pure\nDPO', 'DPO +\nSFT']
    colors = ['#8190a5', '#28875b', '#d19a3a', '#4679ae', '#8564aa']
    panels = [('Correct recommendations', 'correct', 'Higher is better'),
              ('Overpriced Novera selected', 'favored_selected', 'Lower is better; targeted cases only'),
              ('Malformed answers', 'malformed', 'Lower is better'),
              ('Budget or battery violations', 'constraint_violation', 'Lower is better; valid product IDs')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.3))
    fig.subplots_adjust(left=.09, right=.97, bottom=.16, top=.78, hspace=.63, wspace=.26)
    fig.suptitle('Recommendation integrity: held-out final comparison', x=.09, y=.96,
                 ha='left', fontsize=17, fontweight='bold')
    fig.text(.09, .87, '200 parent scenarios × 18 brand/order variants × 5 frozen conditions | Greedy generation, strict scoring', color='#526070')
    for ax, (title, field, note) in zip(axes.flat, panels):
        rows = {r['condition']: r for r in data if r['metric'] == field}
        if len(rows) != 5 or set(rows) != set(names):
            raise ValueError(f'Expected all five conditions for {field}')
        values = [100 * float(rows[n]['rate']) for n in names]
        intervals = [[100 * float(rows[n][key]) for key in
                      ('group_bootstrap_95pct_low', 'group_bootstrap_95pct_high')] for n in names]
        errors = [[max(0., v - ci[0]) for v, ci in zip(values, intervals)],
                  [max(0., ci[1] - v) for v, ci in zip(values, intervals)]]
        ax.bar(range(5), values, color=colors, width=.64, yerr=errors,
               error_kw={'capsize': 3, 'linewidth': 1, 'ecolor': '#344154'})
        for index, (value, interval) in enumerate(zip(values, intervals)):
            ax.text(index, max(value, interval[1]) + 2, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(5), labels)
        ax.set_ylim(0, max(12, min(115, max(c[1] for c in intervals) + 12)))
        ax.set_ylabel('Answers (%)')
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold', pad=23)
        ax.text(0, 1.04, note, transform=ax.transAxes, fontsize=9, color='#526070')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.text(.09, .045, 'Error bars: descriptive 95% parent-group bootstrap intervals; variants are correlated.\n'
             'One model and one seed per recipe. Perfect observations yield degenerate intervals, not population certainty.',
             fontsize=10, color='#526070')
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        path = args.output_directory / f'final_comparison.{extension}'
        fig.savefig(path, dpi=180, facecolor='white')
        if extension == 'svg':
            svg = path.read_text(encoding='utf-8')
            path.write_text('\n'.join(line.rstrip() for line in svg.splitlines()) + '\n',
                            encoding='utf-8', newline='\n')
    print(f'Saved figure to {args.output_directory}')


if __name__ == '__main__':
    main()
