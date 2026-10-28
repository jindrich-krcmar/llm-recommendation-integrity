"""Exercise CUDA forward/backward and bitsandbytes, without model downloads."""
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch


def main():
    assert torch.cuda.is_available(), 'CUDA is unavailable in this Python environment'
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(512, 512, device=device, dtype=torch.bfloat16, requires_grad=True)
    loss = (x @ x.T).float().square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    torch.cuda.synchronize()

    import bitsandbytes as bnb
    layer = bnb.nn.Linear4bit(64, 32, compute_dtype=torch.bfloat16,
                             quant_type='nf4').to(device)
    inputs = torch.randn(2, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    layer(inputs).float().square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    torch.cuda.synchronize()
    report = {
        'python': platform.python_version(),
        'packages': {p: version(p) for p in ['torch', 'transformers', 'peft', 'trl',
                                            'accelerate', 'datasets', 'bitsandbytes']},
        'gpu': torch.cuda.get_device_name(0),
        'cuda_runtime': torch.version.cuda,
        'compute_capability': list(torch.cuda.get_device_capability()),
        'vram_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        'bf16_supported': torch.cuda.is_bf16_supported(),
        'cuda_bf16_forward_backward': 'passed',
        'bitsandbytes_nf4_forward_backward': 'passed',
    }
    print(json.dumps(report, indent=2))
    Path('outputs').mkdir(exist_ok=True)
    Path('outputs/environment-check.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
