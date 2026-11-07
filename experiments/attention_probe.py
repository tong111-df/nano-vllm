"""Exercise real Attention, Triton KV writes and FlashAttention on tiny tensors."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def exercise(report):
    import torch
    from nanovllm.layers.attention import Attention
    from nanovllm.utils.context import set_context, reset_context
    report['import_passed'] = True
    report['versions'] = {name: version(name) for name in ('torch', 'triton', 'flash-attn')}
    report['device'] = torch.cuda.get_device_name(0)
    torch.manual_seed(0)
    heads, kv_heads, dim = 2, 1, 64

    def reference(q, k, v, causal):
        q = q.float().transpose(0, 1)
        k = k.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        v = v.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * dim ** -0.5
        if causal:
            mask = torch.ones(scores.shape[-2:], device='cuda', dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float('-inf'))
        return (scores.softmax(-1) @ v).transpose(0, 1)

    for dtype in (torch.float16, torch.bfloat16):
        layer = Attention(heads, dim, dim ** -0.5, kv_heads).cuda()
        # Deliberately use physical block 1, not 0, to exercise paged indexing.
        layer.k_cache = torch.zeros((2, 256, kv_heads, dim), device='cuda', dtype=dtype)
        layer.v_cache = torch.zeros_like(layer.k_cache)
        q = torch.randn((5, heads, dim), device='cuda', dtype=dtype) * 0.2
        k = torch.randn((5, kv_heads, dim), device='cuda', dtype=dtype) * 0.2
        v = torch.randn_like(k) * 0.2
        tables = torch.tensor([[1]], device='cuda', dtype=torch.int32)
        for stage, start, end in [('prefill', 0, 3), ('prefix_prefill', 3, 4), ('decode', 4, 5)]:
            report['current_stage'] = f'{dtype}/{stage}'
            slots = torch.arange(256 + start, 256 + end, device='cuda', dtype=torch.int32)
            if stage == 'decode':
                set_context(False, slot_mapping=slots, block_tables=tables,
                            context_lens=torch.tensor([end], device='cuda', dtype=torch.int32))
            else:
                set_context(True, slot_mapping=slots,
                            cu_seqlens_q=torch.tensor([0, end - start], device='cuda', dtype=torch.int32),
                            cu_seqlens_k=torch.tensor([0, end], device='cuda', dtype=torch.int32),
                            max_seqlen_q=end - start, max_seqlen_k=end,
                            block_tables=None if start == 0 else tables)
            try:
                with torch.inference_mode():
                    actual = layer(q[start:end], k[start:end], v[start:end]).reshape(end - start, heads, dim)
                    torch.cuda.synchronize()
                    expected = reference(q[start:end], k[:end], v[:end], causal=start == 0)
                    error = (actual.float() - expected).abs().max().item()
                    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
                    torch.testing.assert_close(layer.k_cache[1, :end], k[:end], atol=0, rtol=0)
                    torch.testing.assert_close(layer.v_cache[1, :end], v[:end], atol=0, rtol=0)
                report['checks'].append({'stage': stage, 'dtype': str(dtype),
                                          'max_abs_error': error, 'kv_write_exact': True,
                                          'atol': 0.02, 'rtol': 0.02, 'status': 'passed'})
            finally:
                reset_context()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/unit18/attention_probe')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    report = {'created_at_utc': now.isoformat(), 'python': platform.python_version(),
              'platform': platform.platform(), 'import_passed': False, 'checks': [],
              'status': 'attention_probe_failed',
              'scope': 'Tiny GQA forward cases with FP32 mathematical reference and KV writes; not full model correctness or performance.'}
    try:
        exercise(report)
        report['status'] = 'attention_probe_passed'
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    output = args.output_root / (now.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Saved: {output}')
    return 0 if report['status'] == 'attention_probe_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
