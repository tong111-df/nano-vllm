"""Small CUDA tensor check; success does not mean nano-vLLM inference passed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/unit18/cuda_probe')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    report = {'created_at_utc': now.isoformat(), 'python': platform.python_version(),
              'platform': platform.platform(), 'status': 'cuda_tensor_probe_failed',
              'scope': 'CUDA tensor operations only; not model inference, numerical reference validation or a benchmark.'}
    try:
        import torch
        import torch.distributed as dist
        report.update(torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(),
                      nccl_backend_built=dist.is_nccl_available())
        if not torch.cuda.is_available():
            raise RuntimeError('PyTorch cannot access CUDA')
        torch.cuda.set_device(0)
        report.update(device=torch.cuda.get_device_name(0),
                      device_capability=list(torch.cuda.get_device_capability(0)),
                      compiled_arches=torch.cuda.get_arch_list())
        inputs = torch.ones((64, 64), dtype=torch.float32, device='cuda')
        result = inputs @ inputs
        torch.cuda.synchronize()
        if not (result == 64).all().item():
            raise RuntimeError('CUDA matrix multiplication did not return expected values')
        report.update(status='cuda_tensor_probe_passed', result_device=str(result.device),
                      result_shape=list(result.shape), result_sum=result.sum().item())
    except Exception as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    output = args.output_root / (now.strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    print(f'Saved: {output}')
    return 0 if report['status'] == 'cuda_tensor_probe_passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
