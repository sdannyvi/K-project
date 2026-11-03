"""Exact model-revision access test (WP0): resolve every open model in the
manifest to a Hugging Face commit SHA and confirm this account can download it.

    python -m pnas_pipeline.provenance.revisions [--out reports/model_access.json]
"""
from __future__ import annotations

import argparse
import sys

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from ..io import atomic_write_json
from ..manifest import load_manifest

SKIP_GROUPS = {"negative_closed", "required_revisions"}


def open_models(manifest: dict) -> list[str]:
    return sorted({m for group, ids in manifest["models"].items() if group not in SKIP_GROUPS for m in ids})


def check_model(api: HfApi, model_id: str) -> dict:
    try:
        info = api.model_info(model_id)
    except RepositoryNotFoundError:
        return {"model": model_id, "ok": False, "error": "repository not found"}
    except HfHubHTTPError as exc:
        return {"model": model_id, "ok": False, "error": f"metadata: {exc}"}
    record = {"model": model_id, "sha": info.sha, "gated": bool(info.gated)}
    try:
        hf_hub_download(model_id, "config.json", revision=info.sha)
        record["ok"] = True
    except GatedRepoError:
        record.update(ok=False, error="gated: run `hf auth login` and accept the license on huggingface.co")
    except HfHubHTTPError as exc:
        record.update(ok=False, error=f"download: {exc}")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="cluster/config/experiment_manifest.yaml")
    parser.add_argument("--out", default="reports/model_access.json")
    args = parser.parse_args(argv)
    api = HfApi()
    results = [check_model(api, m) for m in open_models(load_manifest(args.manifest))]
    atomic_write_json(args.out, results)
    for r in results:
        status = f"OK  {r['sha'][:12]}" if r["ok"] else f"FAIL {r['error']}"
        print(f"{r['model']:50s} {status}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
