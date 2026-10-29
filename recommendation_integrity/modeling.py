"""Shared pinned-model loading; imports GPU libraries only when called."""
import hashlib
import os
from pathlib import Path

from .artifacts import ROOT, read_json

SNAPSHOT_FILES = [
    'config.json', 'generation_config.json', 'model*.safetensors',
    'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'vocab.json', 'merges.txt',
    'chat_template.jinja', 'LICENSE', 'README.md',
]


def configure_offline_cache(*, disable_tracking=False):
    os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    if disable_tracking:
        os.environ['DO_NOT_TRACK'] = '1'


def prepare_cuda(seed):
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with BF16 support is required')
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()


def cached_snapshot(config):
    from huggingface_hub import snapshot_download
    snapshot = snapshot_download(
        config['model_id'], revision=config['revision'],
        cache_dir=ROOT / '.cache/huggingface/hub', local_files_only=True,
        token=False, allow_patterns=SNAPSHOT_FILES,
    )
    if Path(snapshot).name != config['revision']:
        raise RuntimeError('Wrong cached model revision')
    return snapshot


def load_tokenizer(snapshot):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise RuntimeError('Missing pinned chat template')
    return tokenizer


def load_base_model(config, snapshot, *, training=False):
    import torch
    from transformers import AutoModelForCausalLM
    dtype = getattr(torch, config['dtype'])
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, dtype=dtype,
        attn_implementation=config['attention_implementation'], device_map={'': 'cuda:0'},
    )
    if not all(p.device.type == 'cuda' and p.dtype == dtype for p in model.parameters()):
        raise RuntimeError('Incorrect dtype or device placement')
    if training:
        model.config.use_cache = False
    else:
        model.eval()
    return model


def load_adapter(model, directory, config):
    from peft import PeftModel
    info = read_json(Path(directory) / 'adapter_config.json')
    if (info['base_model_name_or_path'] != config['model_id'] or
            info.get('revision') != config['revision']):
        raise RuntimeError('Adapter base model or revision does not match the pinned checkpoint')
    return PeftModel.from_pretrained(model, str(directory), is_trainable=False,
                                     local_files_only=True).eval()


def frozen_base_digest(model):
    """Check that LoRA training does not change the frozen base weights."""
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if 'lora_' not in name:
            digest.update(name.encode())
            digest.update(parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()
