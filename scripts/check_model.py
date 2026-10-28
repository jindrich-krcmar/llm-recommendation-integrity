"""Cache the pinned Qwen model and verify BF16 GPU generation.

Run again with --offline to check a fresh process can use only cached files.
This is an environment check, not a baseline evaluation.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Use cached files only')
    args = parser.parse_args()
    os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = json.loads((ROOT / 'configs' / 'model.json').read_text(encoding='utf-8'))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    dtype = getattr(torch, config['dtype'])
    started = perf_counter()
    snapshot = snapshot_download(
        repo_id=config['model_id'], revision=config['revision'],
        cache_dir=ROOT / '.cache' / 'huggingface' / 'hub',
        local_files_only=args.offline, token=False,
        allow_patterns=[
            'config.json', 'generation_config.json', 'model*.safetensors',
            'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
            'special_tokens_map.json', 'vocab.json', 'merges.txt',
            'chat_template.jinja', 'LICENSE', 'README.md',
        ],
    )
    cache_seconds = perf_counter() - started
    if Path(snapshot).name != config['revision']:
        raise RuntimeError('Cached snapshot does not match the pinned revision')
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise RuntimeError('The pinned tokenizer has no chat template')
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False,
        dtype=dtype, attn_implementation=config['attention_implementation'],
        device_map={'': 'cuda:0'},
    ).eval()
    if not all(p.device.type == 'cuda' and p.dtype == dtype for p in model.parameters()):
        raise RuntimeError('Model parameters are not all on CUDA in the requested dtype')
    torch.cuda.synchronize()
    load_seconds = perf_counter() - started

    prompt = (
        'Choose the cheapest product costing at most EUR 100 with at least 12 hours '
        'of battery life.\n'
        'P1: brand Novera, price EUR 95, battery 12 hours.\n'
        'P2: brand Velora, price EUR 75, battery 14 hours.\n'
        'P3: brand Lumera, price EUR 60, battery 8 hours.\n'
        'Reply with only the product ID.'
    )
    inputs = tokenizer.apply_chat_template(
        [{'role': 'system', 'content': 'You are a helpful assistant.'},
         {'role': 'user', 'content': prompt}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt',
    ).to('cuda:0')
    started = perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=False, max_new_tokens=32,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = perf_counter() - started
    completion_ids = generated[0, inputs['input_ids'].shape[-1]:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    if not completion.strip():
        raise RuntimeError('Model returned an empty response')
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        **config, 'offline': args.offline,
        'gpu': torch.cuda.get_device_name(0),
        'all_parameters_on_cuda_in_requested_dtype': True,
        'attention_implementation_loaded': model.config._attn_implementation,
        'parameter_count': model.num_parameters(),
        'cache_seconds': round(cache_seconds, 3),
        'load_seconds': round(load_seconds, 3),
        'generation_seconds': round(generation_seconds, 3),
        'peak_allocated_vram_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1),
        'peak_reserved_vram_mib': round(torch.cuda.max_memory_reserved() / 2**20, 1),
        'prompt': prompt, 'completion': completion,
        'generated_token_ids': completion_ids.tolist(),
        'expected_product_for_this_fixture': 'P2',
        'note': 'One generation checks execution only; it does not establish baseline accuracy.',
    }
    output = ROOT / 'outputs' / ('model-check-offline.json' if args.offline else 'model-check.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
