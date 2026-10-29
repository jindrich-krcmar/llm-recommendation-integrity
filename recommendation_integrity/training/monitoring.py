"""Development validation groups and durable training/evaluation logs."""
import json
from pathlib import Path
from time import perf_counter

from ..data.sft import signature
from ..task import make_cases, scenario_from_dict


def validation_cases(training_cases, development_seed):
    """Use the existing development groups, always with oracle-correct targets."""
    cases = make_cases(development_seed)
    training_specs = {signature(scenario_from_dict(c['scenario'])) for c in training_cases}
    validation_specs = {signature(scenario_from_dict(c['scenario'])) for c in cases}
    if training_specs & validation_specs:
        raise ValueError('Training/validation specification overlap')
    return cases


def append_metrics(directory, step, epoch, logs):
    """Save both loss and eval_loss events without merging their different steps."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps({'step': step, 'epoch': epoch, **logs}) + '\n')


def progress_callback(maximum_seconds, started):
    # Keep CLI help and data tests independent of GPU/training dependencies.
    from transformers import TrainerCallback

    class ProgressMonitor(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if perf_counter() - started > maximum_seconds:
                control.should_training_stop = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            append_metrics(args.output_dir, state.global_step, state.epoch, logs or {})
            # Persist progress at each logged event, not only at training completion.
            state.save_to_json(str(Path(args.output_dir) / 'trainer_state.json'))
            return control

    return ProgressMonitor()
