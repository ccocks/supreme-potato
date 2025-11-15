#!/usr/bin/env python3
"""
DynamoE Merger - Multi-worker model merger with safe 2% jitter
Optimized for low RAM/disk environments (GitHub Actions)
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

# Minimal imports (install with pip)
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
LOCK_FILE = "dynamoe_merger.lock"
STATE_FILE = "dynamoe_merger.state.json"
LOG_FILE = "dynamoe_merger.log"

LOCK_TTL_HOURS = 2  # Shorter for GitHub Actions
MAX_RETRIES = 3  # Less retries for faster failure
BACKOFF_BASE_SEC = 5

ESSENTIAL_SUFFIXES = (".safetensors", ".py", ".bin", ".json", ".model", ".config")
SAFE_SKIP_PATTERNS = (
    "embed", "embedding", "word_embeddings", "lm_head", "final_logits_bias",
    "layernorm", "rms_norm", "norm.", ".ln", "rope", "rotary", "rotary_emb",
    "inv_freq", "alibi", "position", "router", "route", "gate", "gating",
    "expert_score", "topk", "switch", "dispatch", "bias"
)

# GitHub Actions constraints
JITTER_PCT = 0.02
JITTER_MAX_GB = 2.0  # Skip jitter for files > 2GB (Actions has 7GB RAM)
CHUNK_ELEMS = 1_000_000  # Smaller chunks for memory
DEFAULT_BATCH_MAX_GB = 5.0  # Small batches (Actions has 14GB disk)

# ============ SETUP LOGGING ============
def setup_logging(verbose=False):
    logger = logging.getLogger("DynamoE")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# ============ UTILITIES ============
def get_disk_usage():
    """Get available disk space in GB"""
    stat = shutil.disk_usage("/")
    return stat.free / (1024**3)

def cleanup_temp():
    """Aggressive temp cleanup for Actions"""
    temp_dirs = ["/tmp", tempfile.gettempdir()]
    for temp_dir in temp_dirs:
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
    def __init__(self, target_repo, hf_token):
        self.target_repo = target_repo
        self.api = HfApi(token=hf_token)
        self.worker_id = f"worker_{os.environ.get('GITHUB_RUN_ID', 'local')}_{os.getpid()}"
        self.api.create_repo(repo_id=self.target_repo, exist_ok=True, repo_type="model")
        logger.info(f"Initialized {self.worker_id} for {self.target_repo}")
        logger.info(f"Available disk: {get_disk_usage():.1f}GB")

    @backoff
    def acquire_lock(self, force_reclaim_after_hours=LOCK_TTL_HOURS):
        now = datetime.utcnow()
        try:
            p = hf_hub_download(repo_id=self.target_repo, filename=LOCK_FILE, repo_type="model")
            with open(p) as f:
                holder = json.load(f)
            ts = datetime.fromisoformat(holder["ts"])
            age = now - ts
            if age < timedelta(hours=force_reclaim_after_hours):
                raise RuntimeError(f"Lock held by {holder['id']} (age {age})")
            logger.info(f"Reclaiming stale lock (age {age})")
            self.api.delete_file(LOCK_FILE, self.target_repo, repo_type="model")
        except HfHubHTTPError as e:
            if e.response.status_code != 404:
                raise
        except FileNotFoundError:
            pass
        
        payload = {"id": self.worker_id, "ts": now.isoformat()}
        self.api.upload_file(
            path_or_fileobj=json.dumps(payload).encode(),
            path_in_repo=LOCK_FILE,
            repo_id=self.target_repo,
            repo_type="model",
            commit_message=f"lock by {self.worker_id}"
        )
        logger.info("Lock acquired")

    @backoff
    def release_lock(self):
        try:
            self.api.delete_file(LOCK_FILE, self.target_repo, repo_type="model")
            logger.info("Lock released")
        except HfHubHTTPError:
            pass

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

    def build_plan(self, models):
        """Build processing plan"""
        plan = []
        total_shards = 0
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
                        total_shards += 1
            except Exception as e:
                logger.error(f"Cannot scan {repo}: {e}")
        
        # Second pass with correct total
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
                    elif any(fname.endswith(s) for s in ESSENTIAL_SUFFIXES):
                        tgt = f"{expert_dir}/{fname}"
                    else:
                        continue
                    plan.append((repo, fname, tgt, jitter))
            except Exception as e:
                logger.error(f"Cannot access {repo}: {e}")
        
        return plan

    def safe_jitter(self, local_path):
        """Memory-efficient jitter"""
        size_gb = os.path.getsize(local_path) / (1024**3)
        if size_gb > JITTER_MAX_GB:
            logger.warning(f"Skip jitter for {Path(local_path).name} ({size_gb:.1f}GB > {JITTER_MAX_GB}GB)")
            return False
        
        try:
            tensors = load_file(local_path)
            changed = False
            
            for name, t in tensors.items():
                if not t.dtype.is_floating_point or t.ndim < 2:
                    continue
                if any(p in name.lower() for p in SAFE_SKIP_PATTERNS):
                    continue
                
                n = t.numel()
                if n == 0:
                    continue
                
                # Apply jitter to 2% of elements
                k = max(1, int(n * JITTER_PCT))
                indices = torch.randperm(n)[:k]
                
                # Small noise relative to tensor scale
                scale = float(t.abs().max().item()) if t.numel() > 0 else 1.0
                noise = torch.randn(k) * scale * 0.001  # 0.1% of max value
                
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

    def process_batch(self, batch, state):
        """Upload batch of files"""
        if not batch:
            return
        
        total_size = sum(b["size"] for b in batch)
        ops = [CommitOperationAdd(path_in_repo=b["tgt"], path_or_fileobj=b["local"]) for b in batch]
        
        self.api.create_commit(
            repo_id=self.target_repo,
            operations=ops,
            commit_message=f"batch {len(batch)} files ({total_size/(1024**3):.1f}GB) by {self.worker_id}",
            repo_type="model"
        )
        
        for b in batch:
            state["files"][b["tgt"]]["status"] = "done"
            os.remove(b["local"])
        
        self.write_state(state, f"batch done by {self.worker_id}")
        batch.clear()

    def run(self, models, batch_gb=DEFAULT_BATCH_MAX_GB):
        """Main processing loop"""
        self.acquire_lock()
        try:
            state = self.read_state()
            state.setdefault("files", {})
            state.setdefault("weight_map", {})
            
            plan = self.build_plan(models)
            
            # Reconcile with existing files
            try:
                existing = set(self.api.list_repo_files(self.target_repo, repo_type="model"))
                for _, _, tgt, jit in plan:
                    if tgt in existing and state["files"].get(tgt, {}).get("status") != "done":
                        state["files"][tgt] = {"status": "done", "jitter": jit}
                        logger.info(f"Already exists: {tgt}")
            except:
                pass
            
            todo = [p for p in plan if state["files"].get(p[2], {}).get("status") != "done"]
            logger.info(f"Files to process: {len(todo)}/{len(plan)}")
            
            if not todo:
                logger.info("Nothing to do")
                return
            
            batch = []
            batch_size = 0
            batch_limit = int(batch_gb * 1024**3)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                for repo, fname, tgt, jitter in todo:
                    # Check disk space
                    if get_disk_usage() < 2.0:
                        logger.warning("Low disk space, cleaning up...")
                        cleanup_temp()
                        if batch:
                            self.process_batch(batch, state)
                            batch_size = 0
                    
                    logger.info(f"Processing {tgt} from {repo}/{fname}")
                    
                    state["files"].setdefault(tgt, {})["status"] = "pending"
                    self.write_state(state, f"pending {tgt}")
                    
                    try:
                        # Download with resume
                        local_path = hf_hub_download(
                            repo_id=repo,
                            filename=fname,
                            repo_type="model",
                            local_dir=tmpdir,
                            resume_download=True
                        )
                        
                        # Update weight map for safetensors
                        if tgt.endswith(".safetensors"):
                            try:
                                with safe_open(local_path, framework="pt") as f:
                                    for key in f.keys():
                                        if key != "__metadata__":
                                            state["weight_map"][key] = os.path.basename(tgt)
                            except:
                                pass
                            
                            # Apply jitter if requested
                            if jitter:
                                self.safe_jitter(local_path)
                        
                        # Add to batch
                        file_size = os.path.getsize(local_path)
                        
                        # Upload large files immediately
                        if file_size > batch_limit:
                            if batch:
                                self.process_batch(batch, state)
                                batch_size = 0
                            self.process_batch([{"local": local_path, "tgt": tgt, "size": file_size}], state)
                        else:
                            # Check if batch would exceed limit
                            if batch and (batch_size + file_size > batch_limit):
                                self.process_batch(batch, state)
                                batch_size = 0
                            
                            batch.append({"local": local_path, "tgt": tgt, "size": file_size})
                            batch_size += file_size
                    
                    except Exception as e:
                        logger.error(f"Failed processing {tgt}: {e}")
                        state["files"][tgt]["status"] = "failed"
                        self.write_state(state, f"failed {tgt}")
                
                # Final batch
                if batch:
                    self.process_batch(batch, state)
            
            # Create index
            logger.info("Creating index...")
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                json.dump({"metadata": {}, "weight_map": state.get("weight_map", {})}, tmp)
                idx_path = tmp.name
            
            try:
                self.api.upload_file(
                    idx_path,
                    "model.safetensors.index.json",
                    self.target_repo,
                    repo_type="model",
                    commit_message="index"
                )
            finally:
                os.remove(idx_path)
            
            logger.info("Merge complete")
        
        finally:
            self.release_lock()

# ============ CLI ============
def main():
    parser = argparse.ArgumentParser(description="DynamoE Model Merger")
    parser.add_argument("--target", required=True, help="Target repo (user/repo)")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    parser.add_argument("--models", required=True, help="JSON list of models")
    parser.add_argument("--batch-gb", type=float, default=DEFAULT_BATCH_MAX_GB, help="Batch size in GB")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging(args.verbose)
    
    try:
        models = json.loads(args.models)
        merger = DynamoEMerger(args.target, args.token)
        merger.run(models, args.batch_gb)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
