#!/usr/bin/env python3
"""
DynamoE Merger - with DYNAMIC BATCHING and STREAMING JITTER.
Fixes AttributeError in stream_jitter and prevents empty commits.
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import tempfile
import subprocess
import random
from pathlib import Path
from datetime import datetime, timedelta
from functools import wraps

# Minimal imports - numpy is now required for this method
try:
    import torch
    import numpy as np
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    print("Install with: pip install torch safetensors huggingface_hub numpy", file=sys.stderr)
    sys.exit(1)

# ============ CONFIG ============
STATE_FILE = "dynamoe_merger.state.json"
LOG_FILE = "dynamoe_merger.log"
MAX_RETRIES = 5
BACKOFF_BASE_SEC = 10

JITTER_PCT = 0.02
JITTER_MAX_GB = 4.0
CHUNK_ELEMS = 1_000_000

SAFE_SKIP_PATTERNS = (
    "embed", "embedding", "word_embeddings", "lm_head", "final_logits_bias",
    "layernorm", "rms_norm", "norm.", ".ln", "rope", "rotary", "rotary_emb",
    "inv_freq", "alibi", "position", "router", "route", "gate", "gating",
    "expert_score", "topk", "switch", "dispatch", "bias"
)

DTYPE_MAP = {
    "F16": np.float16, "BF16": np.float16, "F32": np.float32, "F64": np.float64,
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "BOOL": np.bool_,
}

# ============ LOGGING ============
def setup_logging(verbose=False):
    logger = logging.getLogger("DynamoE")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# ============ UTILITIES ============
def get_disk_usage_gb():
    try:
        stat = shutil.disk_usage("/")
        return stat.free / (1024**3)
    except Exception:
        return 2.0

def backoff(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        for i in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i == MAX_RETRIES:
                    raise
                wait = BACKOFF_BASE_SEC * (2 ** (i - 1))
                logger.warning(f"{fn.__name__} failed ({i}/{MAX_RETRIES}) -> retry in {wait}s: {e}")
                time.sleep(wait)
    return wrapper

# ============ MERGER CLASS ============
class DynamoEMerger:
    def __init__(self, target_repo, hf_token, worker_id=1, total_workers=1):
        self.target_repo = target_repo
        self.api = HfApi(token=hf_token)
        self.worker_id = int(worker_id)
        self.total_workers = int(total_workers)
        self.api.create_repo(repo_id=self.target_repo, exist_ok=True, repo_type="model")
        logger.info(f"Initialized worker {self.worker_id}/{self.total_workers} for {self.target_repo}")

    @backoff
    def read_state(self):
        try:
            p = hf_hub_download(repo_id=self.target_repo, filename=STATE_FILE, repo_type="model")
            with open(p) as f:
                return json.load(f)
        except (HfHubHTTPError, FileNotFoundError, json.JSONDecodeError):
            return {"files": {}, "weight_map": {}, "source_models": []}

    @backoff
    def write_state(self, state, msg="update"):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump(state, tmp, indent=2)
            tmpname = tmp.name
        try:
            self.api.upload_file(path_or_fileobj=tmpname, path_in_repo=STATE_FILE, repo_id=self.target_repo, repo_type="model", commit_message=msg)
        finally:
            os.remove(tmpname)

    def claim_files(self, plan, state):
        my_files = []
        for i, (repo, fname, tgt, jitter) in enumerate(plan):
            if (i % self.total_workers) + 1 == self.worker_id:
                if state.get("files", {}).get(tgt, {}).get("status") != "done":
                    my_files.append((repo, fname, tgt, jitter))
        return my_files

    def build_plan(self, models):
        plan, total_shards, repo_files_map = [], 0, {}
        for model_config in models:
            repo = list(model_config.keys())[0]
            try:
                files = sorted(self.api.list_repo_files(repo, repo_type="model"))
                repo_files_map[repo] = files
                total_shards += sum(1 for f in files if f.endswith(".safetensors"))
            except Exception as e:
                logger.error(f"Cannot scan {repo}: {e}")
        
        shard_idx = 1
        for expert_idx, model_config in enumerate(models):
            repo, jitter = list(model_config.items())[0]
            expert_dir = f"experts/expert_{expert_idx:03d}"
            for fname in repo_files_map.get(repo, []):
                if fname.endswith((".metal",)) or fname.startswith("."):
                    continue
                if fname.endswith(".safetensors"):
                    tgt = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
                    shard_idx += 1
                else:
                    tgt = f"{expert_dir}/{fname}"
                plan.append((repo, fname, tgt, jitter))
        return plan

    def _should_jitter_param(self, name: str, tensor_info: dict) -> bool:
        dtype_str = tensor_info.get("dtype", "")
        if "F" not in dtype_str:
            return False
        if len(tensor_info.get("shape", [])) < 2:
            return False
        lname = name.lower()
        if any(p in lname for p in SAFE_SKIP_PATTERNS):
            logger.debug(f"Skipping jitter for protected tensor: {name}")
            return False
        return True

    def stream_jitter(self, local_path: str):
        file_size_gb = os.path.getsize(local_path) / (1024**3)
        if file_size_gb > JITTER_MAX_GB:
            logger.warning(f"SKIPPED JITTER: {Path(local_path).name} is {file_size_gb:.1f}GB > {JITTER_MAX_GB}GB limit.")
            return False
        if get_disk_usage_gb() < file_size_gb * 1.1:
            logger.warning(f"SKIPPED JITTER: Not enough disk for temp copy of {Path(local_path).name}.")
            return False

        logger.info(f"Applying streaming jitter to {Path(local_path).name} (one tensor at a time)...")
        temp_jitter_path = local_path + ".jittering"
        shutil.copy2(local_path, temp_jitter_path)
        
        try:
            with safe_open(local_path, framework="pt", device="cpu") as f_read:
                metadata = f_read.metadata()
                if not metadata: raise ValueError("Could not read safetensors metadata.")
            
            changed_tensors = 0
            with open(temp_jitter_path, "r+b") as f_write:
                # FIX: Iterate through metadata.items() correctly
                for name, tensor_info in metadata.items():
                    if name == "__metadata__" or not self._should_jitter_param(name, tensor_info):
                        continue

                    shape, dtype_str, (start, end) = tensor_info['shape'], tensor_info['dtype'], tensor_info['data_offsets']
                    f_write.seek(start)
                    tensor_bytes = f_write.read(end - start)
                    
                    np_dtype = DTYPE_MAP.get(dtype_str)
                    if not np_dtype: continue
                    
                    tensor = torch.from_numpy(np.frombuffer(tensor_bytes, dtype=np_dtype).copy()).view(shape)
                    
                    n = tensor.numel()
                    if n == 0: continue
                    k = max(1, int(n * JITTER_PCT))
                    
                    t_float = tensor.float()
                    indices = torch.randperm(n)[:k]
                    scale = float(t_float.abs().max().item()) if n > 0 else 1.0
                    noise = torch.randn(k) * scale * 0.001
                    
                    t_float.view(-1)[indices] += noise
                    
                    modified_tensor = t_float.to(tensor.dtype)
                    f_write.seek(start)
                    f_write.write(modified_tensor.numpy().tobytes())
                    changed_tensors += 1
                    logger.debug(f"Jittered tensor: {name}")

            if changed_tensors > 0:
                os.replace(temp_jitter_path, local_path)
                logger.info(f"Streaming jitter complete. {changed_tensors} tensors modified.")
                return True
            else:
                logger.info(f"No eligible tensors for jitter found.")
                os.remove(temp_jitter_path)
                return False
                
        except Exception as e:
            logger.error(f"Streaming jitter failed: {e}", exc_info=True)
            if os.path.exists(temp_jitter_path): os.remove(temp_jitter_path)
            return False

    @backoff
    def _commit_batch(self, batch_items, state):
        if not batch_items: return
        total_size = sum(b["size"] for b in batch_items)
        ops = [CommitOperationAdd(path_in_repo=b["tgt"], path_or_fileobj=b["local"]) for b in batch_items]
        
        # FIX: Check for empty operations before committing
        if not ops:
            logger.warning("Attempted to commit an empty batch. Skipping.")
            return

        self.api.create_commit(repo_id=self.target_repo, operations=ops, commit_message=f"batch: {len(ops)} files by worker {self.worker_id}")
        
        for b in batch_items:
            state["files"][b["tgt"]]["status"] = "done"
        self.write_state(state, f"batch done by worker {self.worker_id}")
        
        for b in batch_items:
            os.remove(b["local"])
        batch_items.clear()
        logger.info(f"Committed batch of {len(ops)} files ({total_size/(1024**3):.1f} GB).")

    def run(self, models, disk_fraction=0.5):
        logger.info("Starting work with DYNAMIC BATCHING...")
        state = self.read_state()
        state.setdefault("files", {}); state.setdefault("weight_map", {})
        
        plan = self.build_plan(models)
        
        try:
            remote_files = set(self.api.list_repo_files(self.target_repo, repo_type="model"))
            for _, _, tgt, _ in plan:
                if tgt in remote_files:
                    state["files"].setdefault(tgt, {})["status"] = "done"
        except Exception as e:
            logger.warning(f"Could not reconcile with remote repo: {e}")

        my_files = self.claim_files(plan, state)
        if not my_files:
            logger.info("No new files to process for this worker.")
            if self.worker_id == 1: self.finalize_if_complete(plan, state)
            return

        logger.info(f"Worker {self.worker_id} will process {len(my_files)} files.")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            free_space_gb = get_disk_usage_gb()
            bytes_limit = int(free_space_gb * disk_fraction * (1024**3))
            logger.info(f"Available disk: {free_space_gb:.1f}GB. Dynamic batch limit set to {bytes_limit/(1024**3):.1f}GB.")
            
            batch, batch_bytes = [], 0
            
            for repo, fname, tgt, jitter in my_files:
                try:
                    logger.info(f"Preparing {tgt}...")
                    
                    local_path = hf_hub_download(repo_id=repo, filename=fname, local_dir=tmpdir, resume_download=True)
                    fsize = os.path.getsize(local_path)
                    
                    self.process_file(local_path, tgt, jitter, state)

                    if fsize > bytes_limit:
                        if batch: self._commit_batch(batch, state); batch_bytes=0
                        self.process_and_upload_single(local_path, tgt, state)
                        continue
                    
                    if batch and (batch_bytes + fsize > bytes_limit):
                        self._commit_batch(batch, state)
                        batch_bytes = 0
                    
                    batch.append({"local": local_path, "tgt": tgt, "size": fsize})
                    batch_bytes += fsize
                
                except Exception as e:
                    logger.error(f"Failed processing {tgt}: {e}")
                    state["files"][tgt] = {"status": "failed"}
                    self.write_state(state, f"failed: {tgt}")
            
            if batch:
                self._commit_batch(batch, state)
        
        if self.worker_id == 1:
            self.finalize_if_complete(plan, state)
            
        logger.info(f"Worker {self.worker_id} finished.")

    def process_file(self, local_path, tgt, jitter, state):
        if tgt.endswith(".safetensors"):
            with safe_open(local_path, framework="pt") as f:
                for key in f.keys():
                    if key != "__metadata__":
                        state["weight_map"][key] = os.path.basename(tgt)
            if jitter:
                self.stream_jitter(local_path)
    
    def process_and_upload_single(self, local_path, tgt, state):
        self._commit_batch([{"local": local_path, "tgt": tgt, "size": os.path.getsize(local_path)}], state)

    def finalize_if_complete(self, plan, state):
        if not plan: return
        all_done = all(state.get("files", {}).get(p[2], {}).get("status") == "done" for p in plan)
        if not all_done: return

        logger.info("All files processed. Finalizing index...")
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump({"metadata": {}, "weight_map": state.get("weight_map", {})}, tmp, indent=2)
            idx_path = tmp.name
        try:
            self.api.upload_file(path_or_fileobj=idx_path, path_in_repo="model.safetensors.index.json", repo_id=self.target_repo, repo_type="model", commit_message="Final index")
        finally:
            os.remove(idx_path)
        logger.info("Finalize complete.")

# ============ CLI ============
def main():
    parser = argparse.ArgumentParser(description="DynamoE Merger (Dynamic Batching)")
    parser.add_argument("--target", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--worker-id", type=int, default=1)
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--disk-fraction", type=float, default=0.5)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging(args.verbose)
    
    try:
        models = json.loads(args.models)
        merger = DynamoEMerger(args.target, args.token, args.worker_id, args.total_workers)
        merger.run(models, disk_fraction=args.disk_fraction)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=args.verbose)
        sys.exit(1)

if __name__ == "__main__":
    main()
