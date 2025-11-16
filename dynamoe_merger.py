#!/usr/bin/env python3
"""
DynamoE Merger - TRUE PARALLEL WORKER VERSION
Fix: All HfApi calls use keyword arguments
Optimized for GitHub Actions with limited RAM/disk
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

# Minimal imports
try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch safetensors huggingface_hub")
    sys.exit(1)

# ============ CONFIG ============
STATE_FILE = "dynamoe_merger.state.json"
LOG_FILE = "dynamoe_merger.log"

MAX_RETRIES = 3
BACKOFF_BASE_SEC = 5

JITTER_PCT = 0.02
JITTER_MAX_GB = 2.0  # Actions has 7GB RAM
CHUNK_ELEMS = 1_000_000
DEFAULT_BATCH_MAX_GB = 3.0  # Actions has 14GB disk

SAFE_SKIP_PATTERNS = (
    "embed", "embedding", "word_embeddings", "lm_head", "final_logits_bias",
    "layernorm", "rms_norm", "norm.", ".ln", "rope", "rotary", "rotary_emb",
    "inv_freq", "alibi", "position", "router", "route", "gate", "gating",
    "expert_score", "topk", "switch", "dispatch", "bias"
)

# ============ LOGGING ============
def setup_logging(verbose=False):
    logger = logging.getLogger("DynamoE")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# ============ UTILITIES ============
def get_disk_usage():
    stat = shutil.disk_usage("/")
    return stat.free / (1024**3)

def cleanup_temp():
    for temp_dir in ["/tmp", tempfile.gettempdir()]:
        try:
            for item in Path(temp_dir).iterdir():
                if item.is_dir() and item.name.startswith("tmp"):
                    shutil.rmtree(item, ignore_errors=True)
        except:
            pass

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
        logger.info(f"Available disk: {get_disk_usage():.1f}GB")

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
            # FIXED: Use keyword arguments
            self.api.upload_file(
                path_or_fileobj=tmpname,
                path_in_repo=STATE_FILE,
                repo_id=self.target_repo,
                repo_type="model",
                commit_message=msg
            )
        finally:
            os.remove(tmpname)

    def claim_files(self, plan, state):
        my_files = []
        for i, (repo, fname, tgt, jitter) in enumerate(plan):
            # Assign files round-robin based on worker_id
            if (i % self.total_workers) + 1 == self.worker_id:
                file_state = state["files"].get(tgt, {})
                if file_state.get("status") != "done":
                    # This file is for me and isn't done
                    my_files.append((repo, fname, tgt, jitter))
        return my_files
    
    def build_plan(self, models):
        plan = []
        total_shards = 0
        # Count shards
        for model_config in models:
            repo = list(model_config.keys())[0]
            try:
                files = sorted(self.api.list_repo_files(repo, repo_type="model"))
                total_shards += sum(1 for f in files if f.endswith(".safetensors"))
            except Exception: pass
        # Build plan
        shard_idx = 1
        for expert_idx, model_config in enumerate(models):
            repo = list(model_config.keys())[0]
            jitter = list(model_config.values())[0]
            expert_dir = f"experts/expert_{expert_idx:03d}"
            try:
                files = sorted(self.api.list_repo_files(repo, repo_type="model"))
                for fname in files:
                    if fname.endswith(".metal") or fname.startswith("."): continue
                    if fname.endswith(".safetensors"):
                        tgt = f"model-{shard_idx:05d}-of-{total_shards:05d}.safetensors"
                        shard_idx += 1
                    else:
                        tgt = f"{expert_dir}/{fname}"
                    plan.append((repo, fname, tgt, jitter))
            except Exception as e:
                logger.error(f"Cannot access {repo}: {e}")
        return plan

    def safe_jitter(self, local_path):
        size_gb = os.path.getsize(local_path) / (1024**3)
        if size_gb > JITTER_MAX_GB:
            logger.warning(f"Skip jitter for {Path(local_path).name} ({size_gb:.1f}GB)")
            return False
        try:
            tensors = load_file(local_path)
            changed = False
            for name, t in tensors.items():
                if not t.dtype.is_floating_point or t.ndim < 2: continue
                if any(p in name.lower() for p in SAFE_SKIP_PATTERNS): continue
                n = t.numel()
                k = max(1, int(n * JITTER_PCT))
                indices = torch.randperm(n)[:k]
                scale = float(t.abs().max().item()) if n > 0 else 1.0
                noise = torch.randn(k) * scale * 0.001
                t.view(-1)[indices] += noise.to(t.dtype)
                changed = True
            if changed:
                save_file(tensors, local_path)
                logger.info(f"Jitter applied to {Path(local_path).name}")
            return changed
        except Exception as e:
            logger.error(f"Jitter failed: {e}")
            return False

    def process_batch(self, batch, state):
        if not batch: return
        total_size = sum(b["size"] for b in batch)
        ops = [CommitOperationAdd(path_in_repo=b["tgt"], path_or_fileobj=b["local"]) for b in batch]
        
        # FIXED: Use keyword arguments
        self.api.create_commit(
            repo_id=self.target_repo,
            operations=ops,
            commit_message=f"batch {len(batch)} files ({total_size/(1024**3):.1f}GB) by worker {self.worker_id}",
            repo_type="model"
        )
        
        for b in batch:
            state["files"][b["tgt"]]["status"] = "done"
            os.remove(b["local"])
        
        self.write_state(state, f"batch done by worker {self.worker_id}")
        batch.clear()

    def run(self, models, batch_gb=DEFAULT_BATCH_MAX_GB):
        logger.info("Starting work...")
        state = self.read_state()
        state.setdefault("files", {})
        state.setdefault("weight_map", {})
        
        plan = self.build_plan(models)
        
        # Reconcile with existing remote files
        try:
            remote_files = set(self.api.list_repo_files(self.target_repo, repo_type="model"))
            for _, _, tgt, jit in plan:
                if tgt in remote_files:
                    state["files"].setdefault(tgt, {})["status"] = "done"
        except Exception as e:
            logger.warning(f"Could not reconcile with remote repo: {e}")

        # Claim files
        my_files = self.claim_files(plan, state)
        if not my_files:
            logger.info("No files to process for this worker.")
            # If I'm worker 1, I'll finalize if needed
            if self.worker_id == 1:
                self.finalize_if_complete(plan, state)
            return

        logger.info(f"Worker {self.worker_id} will process {len(my_files)} files.")
        
        bytes_limit = int(max(1.0, batch_gb) * 1024**3)
        batch, batch_bytes = [], 0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for repo, fname, tgt, jitter in my_files:
                logger.info(f"Processing {tgt}")
                
                if get_disk_usage() < 2.0:
                    logger.warning("Low disk space, flushing batch")
                    self.process_batch(batch, state)
                    batch_bytes = 0
                    cleanup_temp()
                
                try:
                    local_path = hf_hub_download(repo_id=repo, filename=fname, local_dir=tmpdir, resume_download=True)
                    if tgt.endswith(".safetensors"):
                        with safe_open(local_path, framework="pt") as f:
                            for key in f.keys():
                                if key != "__metadata__":
                                    state["weight_map"][key] = os.path.basename(tgt)
                        if jitter:
                            self.safe_jitter(local_path)
                    
                    fsize = os.path.getsize(local_path)
                    if fsize > bytes_limit:
                        if batch: self.process_batch(batch, state); batch_bytes = 0
                        self.process_batch([{"local": local_path, "tgt": tgt, "size": fsize}], state)
                    else:
                        if batch and (batch_bytes + fsize > bytes_limit):
                            self.process_batch(batch, state); batch_bytes = 0
                        batch.append({"local": local_path, "tgt": tgt, "size": fsize})
                        batch_bytes += fsize
                
                except Exception as e:
                    logger.error(f"Failed processing {tgt}: {e}")
                    state["files"][tgt]["status"] = "failed"
                    self.write_state(state, f"failed {tgt}")
            
            if batch:
                self.process_batch(batch, state)
        
        if self.worker_id == 1:
            self.finalize_if_complete(plan, state)
            
        logger.info(f"Worker {self.worker_id} finished.")

    def finalize_if_complete(self, plan, state):
        # Check if all files in the plan are done
        all_done = all(state["files"].get(p[2], {}).get("status") == "done" for p in plan)
        if not all_done:
            return

        logger.info("All files are done, finalizing...")
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump({"metadata": {}, "weight_map": state.get("weight_map", {})}, tmp)
            idx_path = tmp.name
        try:
            # FIXED: Use keyword arguments
            self.api.upload_file(
                path_or_fileobj=idx_path,
                path_in_repo="model.safetensors.index.json",
                repo_id=self.target_repo,
                repo_type="model",
                commit_message="Final index"
            )
        finally:
            os.remove(idx_path)
        logger.info("Finalize complete.")

# ============ CLI ============
def main():
    parser = argparse.ArgumentParser(description="DynamoE Model Merger (Parallel Workers)")
    parser.add_argument("--target", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--worker-id", type=int, default=1)
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--batch-gb", type=float, default=DEFAULT_BATCH_MAX_GB)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging(args.verbose)
    
    try:
        models = json.loads(args.models)
        merger = DynamoEMerger(args.target, args.token, args.worker_id, args.total_workers)
        merger.run(models, args.batch_gb)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
