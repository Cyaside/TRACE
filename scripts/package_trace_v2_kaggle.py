"""Build a minimal private Kaggle input dataset and self-contained script job.

Does not upload, authenticate, or train. No .local evidence, secrets, previous
weights or optimizer snapshots are included in the source allowlist.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import zipfile

from src.common import ROOT, digest, object_hash, read_json, write_json

SOURCE_FILES = (
    "src/__init__.py", "src/common.py", "src/data/__init__.py", "src/data/audit.py", "src/data/download.py",
    "src/features/__init__.py", "src/features/temporal.py", "src/features/v2_dataset.py",
    "src/models/__init__.py", "src/models/neural.py", "src/models/trace_v2.py",
    "src/training/__init__.py", "src/training/checkpoint.py", "src/training/kaggle_runtime.py", "src/training/trace_v2.py",
    "src/evaluation/__init__.py", "src/evaluation/evaluate.py", "configs/trace-v2-kaggle.json")


def build(destination, username, prepared_root=None):
    destination = Path(destination)
    prepared = Path(prepared_root) if prepared_root else ROOT / "data/prepared/teacher-10000-v1"
    data_dir, job_dir = destination / "data", destination / "job"
    data_dir.mkdir(parents=True, exist_ok=True)
    job_dir.mkdir(parents=True, exist_ok=True)
    config = read_json(ROOT / "configs/trace-v2-kaggle.json")
    manifest = read_json(prepared / "manifest.json")
    if manifest["data_hash"] != config["prepared_data_hash"]:
        raise ValueError("Frozen prepared data mismatch")
    for name, expected in manifest["files"].items():
        if digest(prepared / name) != expected:
            raise ValueError(f"Dataset mismatch: {name}")
    source_hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    source_hash = object_hash(source_hashes)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCE_FILES:
            info = zipfile.ZipInfo(name, (2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (ROOT / name).read_bytes())
    with zipfile.ZipFile(data_dir / "prepared.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ["manifest.json", *manifest["files"]]:
            archive.write(prepared / name, name)
    data_archive_hash = digest(data_dir / "prepared.zip")
    (data_dir / "DATA_LICENSE.txt").write_text(
        "KuaiSAR-small, Zhongxiang Sun, Zihua Si, Xiaoxue Zang, Dewei Leng, Yanan Niu, Yang Song, Xiao Zhang, Jun Xu.\n"
        "KuaiSAR: A Unified Search And Recommendation Dataset, CIKM 2023.\n"
        "Source: https://kuaisar.github.io/ and https://zenodo.org/records/8181109\n"
        "License: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
        "Modified: deterministic 10k cohort, chronological request/block preparation, train-only vocabulary.\n"
        "Private research training input. Not a public redistribution or production dataset.\n", encoding="utf-8")
    slug = "trace-v2-inputs-20261010"
    dataset_id = f"{username}/{slug}"
    write_json(data_dir / "dataset-metadata.json", {"id": dataset_id, "title": "TRACE V2 Inputs 20261010",
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "description": "Private deterministic KuaiSAR-small 10k training inputs. Source and attribution in DATA_LICENSE.txt. "
                       "Prepared requests, histories and static caption metadata; original released IDs remain anonymized. "
                       "Derived preparation under CC BY-NC-SA 4.0. https://kuaisar.github.io/"})
    bootstrap = '''"""Private TRACE V2 GPU run. Setup -> measured early updates -> training."""
import base64, hashlib, importlib.metadata, importlib.util, json, os, pathlib, subprocess, sys, time, traceback, zipfile, io
START = time.perf_counter()
SOURCE_HASH = __SOURCE_HASH__
SOURCE_HASHES = __SOURCE_HASHES__
DATA_ARCHIVE_HASH = __DATA_ARCHIVE_HASH__
DATA_MANIFEST_HASH = __DATA_MANIFEST_HASH__
DATA_HASH = __DATA_HASH__
BUNDLE = __BUNDLE__
ROOT = pathlib.Path('/kaggle/working/trace-source')
OUTPUT = pathlib.Path('/kaggle/working/trace-output')
ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
PHASE = 'source_verification'
def save(name, value):
    path = OUTPUT / name
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\\n')
try:
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))) as archive:
        archive.extractall(ROOT)
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Source hash mismatch: ' + name)
    sys.path.insert(0, str(ROOT))
    PHASE = 'input_verification'
    from src.training.kaggle_runtime import resolve_prepared
    data = resolve_prepared('/kaggle/input', manifest_hash=DATA_MANIFEST_HASH,
        data_hash=DATA_HASH, archive_hash=DATA_ARCHIVE_HASH,
        extract_root='/kaggle/working/trace-data')
    print(json.dumps({'phase': PHASE, 'status': 'passed', 'data_root': str(data)}), flush=True)
    PHASE = 'dependencies'
    dependency_start = time.perf_counter()
    # Preserve the platform's scientific/CUDA stack. Installing older NumPy or
    # Arrow globally breaks unrelated image packages, especially on Python 3.13.
    # Only absent libraries are installed into a separate directory, without
    # changing dependencies already supplied by Kaggle.
    requirements = {'numpy': ('numpy', 'numpy>=2.1,<3' if sys.version_info >= (3,13) else 'numpy>=1.26,<3'),
        'pyarrow': ('pyarrow', 'pyarrow>=19,<24'), 'duckdb': ('duckdb', 'duckdb>=1.2.2,<2'),
        'scikit-learn': ('sklearn', 'scikit-learn>=1.6,<2'), 'psutil': ('psutil', 'psutil>=7,<8')}
    missing = [requirement for module, requirement in requirements.values() if importlib.util.find_spec(module) is None]
    if missing:
        target = pathlib.Path('/kaggle/working/trace-extra-dependencies')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
            '--only-binary=:all:', '--no-deps', '--target', str(target), *missing], check=True, timeout=300)
        sys.path.insert(0, str(target))
    dependency_seconds = time.perf_counter() - dependency_start
    PHASE = 'runtime_preflight'
    import torch
    import numpy as np
    import pyarrow as pa
    import duckdb, sklearn, psutil
    if not torch.cuda.is_available():
        raise RuntimeError('Requested GPU unavailable; no CPU training fallback')
    # Exercise the actual NumPy/PyTorch/Arrow boundary, not just package names.
    values = np.asarray([1., 2.], dtype=np.float32)
    np.testing.assert_array_equal(torch.from_numpy(values).to('cuda:0').cpu().numpy(), values)
    np.testing.assert_array_equal(pa.array(values).to_numpy(), values)
    runtime = {'source_hash': SOURCE_HASH, 'dependency_setup_seconds': dependency_seconds,
        'installed_missing_requirements': missing, 'data_root': str(data), 'data_files_verified': True,
        'setup_before_data_reader_seconds': time.perf_counter() - START,
        'python': sys.version, 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'packages': {name: importlib.metadata.version(name) for name in requirements}}
    save('setup.json', runtime)
    print(json.dumps({'phase': PHASE, 'status': 'passed', **runtime}), flush=True)
    PHASE = 'data_reader'
    from src.features.v2_dataset import V2Dataset
    from src.training.trace_v2 import train
    config = json.loads((ROOT / 'configs/trace-v2-kaggle.json').read_text())
    tick = time.perf_counter()
    dataset = V2Dataset(data, cache_size=config['block_cache_size'])
    if dataset.data_hash != config['prepared_data_hash']:
        raise ValueError('Frozen dataset identity mismatch')
    save('data-profile.json', {'load_verify_fit_seconds': time.perf_counter() - tick,
        'arrow_bytes': dataset.rows.nbytes + dataset.blocks.nbytes + dataset.metadata.nbytes,
        'train_search_groups': len(dataset.indexes[('train','search')]), 'train_rec_exposures': dataset.rec_count,
        'test_outcomes_loaded': False})
    print(json.dumps({'setup': 'complete', 'elapsed_seconds': time.perf_counter() - START}), flush=True)
    PHASE = 'training'
    report = train(dataset, config, OUTPUT, source_hash=SOURCE_HASH, job_start=START)
    save('job-result.json', report)
    print(json.dumps({'job_status': report['status'], 'completed_steps': report['completed_steps']}), flush=True)
except BaseException as error:
    save('job-failure.json', {'error_type': type(error).__name__, 'error': str(error),
                            'phase': PHASE, 'traceback': traceback.format_exc(),
                            'elapsed_seconds': time.perf_counter() - START})
    raise
'''
    bootstrap = (bootstrap.replace("__SOURCE_HASH__", repr(source_hash))
                 .replace("__SOURCE_HASHES__", repr(source_hashes))
                 .replace("__DATA_ARCHIVE_HASH__", repr(data_archive_hash))
                 .replace("__DATA_MANIFEST_HASH__", repr(digest(prepared / "manifest.json")))
                 .replace("__DATA_HASH__", repr(manifest["data_hash"]))
                 .replace("__BUNDLE__", repr(base64.b64encode(buffer.getvalue()).decode())))
    (job_dir / "job.py").write_text(bootstrap, encoding="utf-8")
    compile(bootstrap, str(job_dir / "job.py"), "exec")
    kernel_id = f"{username}/trace-v2-10k-training"
    write_json(job_dir / "kernel-metadata.json", {"id": kernel_id, "title": "TRACE V2 10k Training",
        "code_file": "job.py", "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [dataset_id], "competition_sources": [], "kernel_sources": [], "model_sources": []})
    report = {"source_hash": source_hash, "source_files": source_hashes, "config_hash": object_hash(config),
              "data_hash": manifest["data_hash"], "data_archive_sha256": data_archive_hash,
              "data_manifest_sha256": digest(prepared / "manifest.json"),
              "bootstrap_sha256": digest(job_dir / "job.py"),
              "dataset_id": dataset_id, "kernel_id": kernel_id, "private": True,
              "archive_bytes": (data_dir / "prepared.zip").stat().st_size,
              "kernel_url": f"https://www.kaggle.com/code/{kernel_id}", "submitted": False}
    write_json(destination / "bundle-manifest.json", report)
    print(json.dumps({k: report[k] for k in ("source_hash", "dataset_id", "kernel_id", "archive_bytes", "submitted")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--prepared")
    args = parser.parse_args()
    build(args.destination, args.username, args.prepared)
