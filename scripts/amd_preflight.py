"""Check the allocated ROCm device and sparse state-gradient path, without data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch


def checked_csr(crow, col, values, device):
    """Validate matching index dtypes and sorted columns on CPU before transfer."""
    crow = torch.tensor(crow, dtype=torch.int64)
    col = torch.tensor(col, dtype=torch.int64)
    values = torch.tensor(values, dtype=torch.float32)
    assert crow.dtype == col.dtype == torch.int64
    # This catches invalid structure before any rocSPARSE kernel receives it.
    torch.sparse_csr_tensor(crow, col, values, size=(3, 3), check_invariants=True)
    return torch.sparse_csr_tensor(crow.to(device), col.to(device), values.to(device),
                                   size=(3, 3), device=device, check_invariants=True)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def emit(event, **values):
    print(json.dumps({'event': event, **values}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0', choices=('cuda:0', 'cpu'),
                   help='CPU is a local logic check; default requires one allocated ROCm GPU.')
    args = p.parse_args()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == 'cuda':
        assert torch.version.hip, 'Require a ROCm PyTorch runtime'
        assert torch.cuda.is_available(), 'Allocated GPU unavailable to PyTorch'
        assert torch.cuda.device_count() == 1, 'This experiment must see exactly one GPU'
        prop = torch.cuda.get_device_properties(device)
        free, total = torch.cuda.mem_get_info(device)
    else:
        prop, free, total = None, None, None
    runtime = dict(host=platform.node(), job_id=os.environ.get('SLURM_JOB_ID'),
                   python=platform.python_version(), torch=torch.__version__,
                   hip=torch.version.hip, numpy=np.__version__, gpu=prop.name if prop else None,
                   device=str(device), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   total_bytes=total, free_bytes_before=free,
                   device_count=torch.cuda.device_count() if device.type == 'cuda' else 0,
                   sparse_index_dtype='torch.int64', invariant_checks=True)
    emit('runtime', **runtime)
    # Independent tiny dense oracle also distinguishes basic device/GEMM
    # failures from CSR construction or rocSPARSE state-gradient failures.
    emit('dense_device_check_begin')
    dense = torch.tensor([[.2, 0., -.3], [0., .4, 0.], [.5, 0., -.6]], device=device)
    x = torch.tensor([[.3, .2], [.7, -.4], [.1, .8]], device=device,
                     requires_grad=True)
    expected_y = dense @ x.detach()
    expected_dx = dense.T @ (2 * expected_y)
    synchronize(device)
    assert torch.isfinite(expected_y).all()
    emit('dense_device_check_passed')
    emit('checked_csr_construction_begin')
    sparse = checked_csr([0, 2, 3, 5], [0, 2, 1, 0, 2], [.2, -.3, .4, .5, -.6], device)
    synchronize(device)
    emit('checked_csr_construction_passed', crow_dtype=str(sparse.crow_indices().dtype),
         col_dtype=str(sparse.col_indices().dtype), sorted_distinct_columns=True)
    emit('sparse_forward_begin')
    y = torch.sparse.mm(sparse, x)
    synchronize(device)
    torch.testing.assert_close(y, expected_y)
    emit('sparse_forward_passed')
    emit('sparse_state_backward_begin')
    y.square().sum().backward()
    synchronize(device)
    torch.testing.assert_close(x.grad, expected_dx)
    emit('sparse_state_backward_passed')
    runtime['small_sparse_forward_backward'] = 'passed'
    # The model uses an explicitly constructed CSR transpose for backward.
    emit('explicit_transpose_begin')
    transpose = checked_csr([0, 2, 3, 5], [0, 2, 1, 0, 2], [.2, .5, .4, -.3, -.6], device)
    back = torch.sparse.mm(transpose, 2 * expected_y)
    synchronize(device)
    torch.testing.assert_close(back, expected_dx)
    runtime['explicit_csr_transpose_backward'] = 'passed'
    runtime['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(runtime, indent=2) + '\n')
    emit('preflight_complete', **runtime)


if __name__ == '__main__':
    main()
