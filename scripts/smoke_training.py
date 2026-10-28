"""Offline integration check using a tiny, randomly initialized Qwen model.

This validates software plumbing, not recommendation quality or full-model memory.
"""
import copy
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


def main():
    assert torch.cuda.is_available(), 'This check requires CUDA'
    torch.manual_seed(42)
    words = ['[PAD]', '[EOS]', '[UNK]', 'Choose', 'the', 'cheapest', 'product',
             'P1', 'P2', 'P3', '.', ':']
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]',
                                       eos_token='[EOS]', unk_token='[UNK]')
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(words), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
        attention_dropout=0.0, use_cache=False,
    )).to(device='cuda', dtype=torch.bfloat16)
    model.config._attn_implementation = 'sdpa'
    common = dict(per_device_train_batch_size=1, max_steps=2, learning_rate=1e-3,
                  bf16=True, report_to='none', save_strategy='no', logging_steps=1,
                  dataloader_num_workers=0, dataloader_pin_memory=False,
                  gradient_checkpointing=False, optim='adamw_torch', seed=42,
                  disable_tqdm=True)
    sft = SFTTrainer(
        model=model, processing_class=tokenizer,
        peft_config=LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'],
                               lora_dropout=0.0, task_type='CAUSAL_LM'),
        args=SFTConfig(output_dir='outputs/smoke-sft', max_length=64,
                       packing=False, **common),
        train_dataset=Dataset.from_dict({'text': ['Choose the cheapest product : P2 .'] * 2}),
    )
    before = {n: p.detach().clone() for n, p in sft.model.named_parameters() if p.requires_grad}
    sft_result = sft.train()
    assert math.isfinite(sft_result.training_loss)
    assert any(not torch.equal(before[n], p) for n, p in sft.model.named_parameters() if n in before)
    sft.save_model('outputs/smoke-sft/adapter')

    # Merge this tiny adapter so both independent models include the SFT update.
    # The real experiment must separately verify its checkpoint/reference loading.
    policy = sft.model.merge_and_unload()
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    assert all(torch.equal(p, r) for p, r in zip(policy.parameters(), reference.parameters()))
    reference_before = {n: p.detach().clone() for n, p in reference.named_parameters()}
    dpo = DPOTrainer(
        model=policy, ref_model=reference, processing_class=tokenizer,
        peft_config=LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj', 'v_proj'],
                               lora_dropout=0.0, task_type='CAUSAL_LM'),
        args=DPOConfig(output_dir='outputs/smoke-dpo', max_length=64, **common),
        train_dataset=Dataset.from_dict({
            'prompt': ['Choose the cheapest product : '] * 2,
            'chosen': ['P2 .'] * 2, 'rejected': ['P1 .'] * 2,
        }),
    )
    dpo_before = {n: p.detach().clone() for n, p in dpo.model.named_parameters() if p.requires_grad}
    dpo_result = dpo.train()
    assert math.isfinite(dpo_result.training_loss)
    assert any(not torch.equal(dpo_before[n], p) for n, p in dpo.model.named_parameters() if n in dpo_before)
    assert all(not p.requires_grad and torch.equal(p, reference_before[n])
               for n, p in reference.named_parameters())
    dpo.save_model('outputs/smoke-dpo/adapter')
    report = {'test_model': 'tiny random Qwen2 (no pretrained weights)',
              'sft_steps': sft_result.global_step, 'sft_loss': sft_result.training_loss,
              'dpo_steps': dpo_result.global_step, 'dpo_loss': dpo_result.training_loss,
              'sft_and_dpo_adapter_updates': 'passed',
              'dpo_reference_includes_sft_and_stays_frozen': 'passed',
              'peak_allocated_vram_mib': round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    Path('outputs/training-smoke-check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
