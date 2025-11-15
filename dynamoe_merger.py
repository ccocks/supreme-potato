#!/usr/bin/env python3
"""
DynamoE Merger - TRUE PARALLEL WORKER VERSION
No global lock - workers claim individual files via state file
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
    sys.exit(1)

# ============ CONFIG ============
STATE_FILE = "dynamoe_merger.state.json"
LOG_FILE = "dynamoe_merger.log"

MAX_RETRIES = 3
BACKOFF_BASE_SEC = 5

# GitHub Actions constraints
JITTER_PCT = 0.02
JITTER_MAX_GB = 2.0  # Skip jitter for files > 2GB
CHUNK_ELEMS = 1_000_000
DEFAULT_BATCH_MAX_GB = 3.0  # Small batches for Actions

# ============ LOGGING ============
def setup_logging(verbose=False):
    logger = logging.getLogger("DynamoE")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

logger = setup_logging()

def get_disk_usage():
    """Get available disk space in GB"""
    stat = shutil.disk_usage("/")
    return stat.free / (1024**3)

def cleanup_temp():
    """Aggressive cleanup for Actions"""
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
            self.api.upload_file(tmpname, STATE_FILE, self.target_repo, repo_type="model", commit_message=msg)
        finally:
            os.remove(tmpname)

    def claim_files(self, plan, state):
        """Claim unassigned files for this worker"""
        claimed = []
        for repo, fname, tgt, jitter in plan:
            file_state = state["files"].get(tgt, {})
            if file_state.get("status") == "done":
                continue  # Already done
            if file_state.get("assigned_to") is None:
                # File is free - claim it
                state["files"][tgt] = {
                    "status": "pending",
                    "assigned_to": self.worker_id,
                    "src": f"{repo}/{fname}",
                    "jitter": jitter
                }
                claimed.append((repo, fname, tgt, jitter))
        return claimed

    def build_plan(self, models):
        """Build deterministic plan"""
        plan = []
        total_shards = 0
        shard_idx = 1
        
        # Count shards first
        for model_config in models:
            repo = list(model_config.keys())[0]
            try:
                files = sorted(self.api.list_repo_files(repo, repo_type="model"))
                total_shards += sum(1 for f in files if f.endswith(".safetensors"))
            except:
                pass
        
        # Build plan
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

    def run(self, models, batch_gb=DEFAULT_BATCH_MAX_GB):
        """Main processing loop - NO GLOBAL LOCK"""
        logger.info("Starting work (no global lock - claiming individual files)")
        
        state = self.read_state()
        state.setdefault("files", {})
        state.setdefault("weight_map", {})
        state["source_models"] = [list(m.keys())[0] for m in models]
        
        plan = self.build_plan(models)
        
        # Claim files for this worker
        claimed = self.claim_files(plan, state)
        if not claimed:
            logger.info("No files to claim - either all done or claimed by other workers")
            return
        
        logger.info(f"Claimed {len(claimed)} files for worker {self.worker_id}")
        self.write_state(state, f"claimed {len(claimed)} for worker {self.worker_id}")
        
        bytes_limit = int(max(1.0, batch_gb) * (1024**3))
        batch = []
        batch_bytes = 0
        
        with tempfile.TemporaryDirectory() as tmpdir:
            for repo, fname, tgt, jitter in claimed:
                logger.info(f"Processing {tgt} from {repo}/{fname}")
                
                if get_disk_usage() < 2.0:
                    logger.warning("Low disk space, flushing batch")
                    self.process_batch(batch, state)
                    batch_bytes = 0
                    cleanup_temp()
                
                try:
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
                    
                    # Add to batch
                    fsize = os.path.getsize(local_path)
                    if fsize > bytes_limit:
                        # Large file - upload immediately
                        if batch:
                            self.process_batch(batch, state)
                            batch_bytes = 0
                        self.process_batch([{"local": local_path, "tgt": tgt, "size": fsize}], state)
                    else:
                        if batch and (batch_bytes + fsize > bytes_limit):
                            self.process_batch(batch, state)
                            batch_bytes = 0
                        batch.append({"local": local_path, "tgt": tgt, "size": fsize})
                        batch_bytes += fsize
                
                except Exception as e:
                    logger.error(f"Failed processing {tgt}: {e}")
                    state["files"][tgt]["status"] = "failed"
                    self.write_state(state, f"failed {tgt}")
            
            # Final batch
            if batch:
                self.process_batch(batch, state)
        
        # Create index
        logger.info("Creating final index...")
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            json.dump({"metadata": {}, "weight_map": state.get("weight_map", {})}, tmp)
            idx_path = tmp.name
        
        try:
            self.api.upload_file(
                idx_path,
                "model.safetensors.index.json",
                self.target_repo,
                repo_type="model",
                commit_message="Final index"
            )
        finally:
            os.remove(idx_path)
        
        logger.info(f"Worker {self.worker_id} finished")

# ============ CLI ============
def main():
    parser = argparse.ArgumentParser(description="DynamoE Model Merger (Parallel Workers)")
    parser.add_argument("--target", required=True, help="Target repo (user/repo)")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--models", required=True, help="JSON list of models")
    parser.add_argument("--worker-id", type=int, default=1, help="Worker ID (1-based)")
    parser.add_argument("--total-workers", type=int, default=1, help="Total number of workers")
    parser.add_argument("--batch-gb", type=float, default=DEFAULT_BATCH_MAX_GB, help="Batch size in GB")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        global logger
        logger = setup_logging(verbose=True)
    
    try:
        models = json.loads(args.models)
        merger = DynamoEMerger(args.target, args.token, args.worker_id, args.total_workers)
        merger.run(models, args.batch_gb)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
