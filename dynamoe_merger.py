#!/usr/bin/env python3
"""
DynamoE Merger - DISK-SPACE OPTIMIZED
Processes files ONE AT A TIME, immediate cleanup
Optimized for GitHub Actions (14GB disk, 7GB RAM)
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

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

# ============ CONFIG ============
STATE_FILE = "dynamoe_merger.state.json"
LOG_FILE = "dynamoe_merger.log"

MAX_RETRIES = 3
BACKOFF_BASE_SEC = 5

JITTER_PCT = 0.02
JITTER_MAX_GB = 1.0  # Skip jitter for >1GB files
CHUNK_ELEMS = 500_000

# GitHub Actions has ~14GB total, ~10GB available
# Keep at least 2GB free at all times
MIN_FREE_DISK_GB = 2.0

# Process files ONE AT A TIME (no batching for disk space)
PROCESS_ONE_AT_A_TIME = True

def setup_logging(verbose=False):
    logger = logging.getLogger("DynamoE")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    if logger.handlers: logger.handlers.clear()
    logger.addHandler(handler)
    return logger

logger = setup_logging()

def get_disk_usage():
    stat = shutil.disk_usage("/")
    return stat.free / (1024**3)

def cleanup_temp():
    for temp_dir in ["/tmp", tempfile.gettempdir()]:
        try:
            for item in Path(temp_dir).iterdir():
                if item.is_dir() and item.name.startswith("tmp"):
                    shutil.rmtree(item, ignore_errors=True)
        except: pass

def backoff(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        for i in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i == MAX_RETRIES: raise
                wait = BACKOFF_BASE_SEC * (2 ** (i - 1))
                logger.warning(f"{fn.__name__} failed ({i}/{MAX_RETRIES}) -> retry in {wait}s: {e}")
                time.sleep(wait)
    return wrapper

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
        """Claim unassigned files for this worker using round-robin"""
        my_files = []
        for i, (repo, fname, tgt, jitter) in enumerate(plan):
            if (i % self.total_workers) + 1 == self.worker_id:
                file_state = state["files"].get(tgt, {})
                if file_state.get("status") != "done":
                    # Mark as assigned to this worker
                    state["files"][tgt] = {
                        "status": "pending",
                        "assigned_to": self.worker_id,
                        "src": f"{repo}/{fname}",
                        "jitter": jitter
                    }
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
            except:
                pass
        
        # Build plan
        shard_idx = 1
        for expert_idx, model_config in enumerate(models):
            repo = list(model_config.keys())[0]
            jitter = list(model_config.values())[0]
            expert_dir = f"experts/expert_{expert_idx:03d}"
            try:
                files = sorted(self.api.list_repo_files(repo, repo_type="model"))
                for fname in files:
                    if fname.endswith(".metal") or fname.startswith("."):
                        continue
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
        """Memory-efficient jitter"""
        size_gb = os.path.getsize(local_path) / (1024**3)
        if size_gb > JITTER_MAX_GB:
            logger.warning(f"Skip jitter for {Path(local_path).name} ({size_gb:.1f}GB)")
            return False
        
        try:
            tensors = load_file(local_path)
            changed = False
            
            for name, t in tensors.items():
                if not t.dtype.is_floating_point or t.ndim < 2:
                    continue
                if any(p in name.lower() for p in SAFE_SKIP_PATTERNS):
                    continue
                
                # Apply small perturbation
                n = t.numel()
                k = max(1, int(n * JITTER_PCT))
                indices = torch.randperm(n)[:k]
                
                # Scale noise to tensor magnitude
                scale = float(t.abs().max().item()) if t.numel() > 0 else 1.0
                noise = torch.randn(k) * scale * 0.001
                
                flat = t.view(-1)
                flat[indices] += noise.to(t.dtype)
                changed = True
            
            if changed:
                save_file(tensors, local_path)
                logger.info(f"Jitter applied to {Path(local_path).name}")
            return changed
        except Exception as e:
            logger.error(f"Jitter failed: {e}")
            return False

    def process_single_file(self, file_info, state):
        """Process and upload ONE file at a time"""
        repo, fname, tgt, jitter = file_info
        
        # Create temp dir for this single file
        tmpdir = tempfile.mkdtemp()
        local_path = None
        
        try:
            # Check disk space before download
            if get_disk_usage() < MIN_FREE_DISK_GB:
                logger.warning(f"Low disk space ({get_disk_usage():.1f}GB), cleaning up...")
                cleanup_temp()
                if get_disk_usage() < MIN_FREE_DISK_GB:
                    logger.error("Still low on disk space after cleanup, aborting")
                    sys.exit(1)
            
            # Download
            logger.info(f"Downloading {repo}/{fname}")
            local_path = hf_hub_download(
                repo_id=repo,
                filename=fname,
                repo_type="model",
                local_dir=tmpdir,
                resume_download=True
            )
            
            # Update weight map
            if tgt.endswith(".safetensors"):
                try:
                    with safe_open(local_path, framework="pt") as f:
                        for key in f.keys():
                            if key != "__metadata__":
                                state["weight_map"][key] = os.path.basename(tgt)
                except:
                    pass
                
                # Apply jitter
                if jitter:
                    self.safe_jitter(local_path)
            
            # Upload immediately (no batching)
            logger.info(f"Uploading {tgt}")
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=tgt,
                repo_id=self.target_repo,
                repo_type="model",
                commit_message=f"upload {tgt} by worker {self.worker_id}"
            )
            
            # Mark as done
            state["files"][tgt]["status"] = "done"
            self.write_state(state, f"done {tgt}")
            
            # Delete local file immediately
            os.remove(local_path)
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.info(f"Completed and cleaned up {tgt}")
            
        except Exception as e:
            logger.error(f"Failed processing {tgt}: {e}")
            state["files"][tgt]["status"] = "failed"
            self.write_state(state, f"failed {tgt}")
            # Cleanup on failure
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
            shutil.rmtree(tmpdir, ignore_errors=True)

    def run(self, models, batch_gb=DEFAULT_BATCH_MAX_GB):
        """Main processing loop - disk-space optimized"""
        logger.info("Starting work (disk-space optimized mode)")
        
        state = self.read_state()
        state.setdefault("files", {})
        state.setdefault("weight_map", {})
        
        plan = self.build_plan(models)
        
        # Reconcile with existing files
        try:
            remote_files = set(self.api.list_repo_files(self.target_repo, repo_type="model"))
            for _, _, tgt, _ in plan:
                if tgt in remote_files:
                    state["files"].setdefault(tgt, {})["status"] = "done"
        except:
            pass

        # Claim files for this worker
        my_files = self.claim_files(plan, state)
        if not my_files:
            logger.info("No files to process for this worker.")
            if self.worker_id == 1:
                self.finalize_if_complete(plan, state)
            return

        logger.info(f"Worker {self.worker_id} will process {len(my_files)} files.")
        
        # Process files ONE AT A TIME (no batching)
        for file_info in my_files:
            self.process_single_file(file_info, state)
        
        # Finalize if I'm worker 1 and all done
        if self.worker_id == 1:
            self.finalize_if_complete(plan, state)
        
        logger.info(f"Worker {self.worker_id} finished")

    def finalize_if_complete(self, plan, state):
        all_done = all(state["files"].get(p[2], {}).get("status") == "done" for p in plan)
        if not all_done:
            return

        logger.info("All files done, finalizing...")
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump({"metadata": {}, "weight_map": state.get("weight_map", {})}, tmp)
            idx_path = tmp.name
        
        try:
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
    parser = argparse.ArgumentParser(description="DynamoE Model Merger (Disk-Optimized)")
    parser.add_argument("--target", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--worker-id", type=int, default=1)
    parser.add_argument("--total-workers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging(args.verbose)
    
    try:
        models = json.loads(args.models)
        merger = DynamoEMerger(args.target, args.token, args.worker_id, args.total_workers)
        merger.run(models)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
