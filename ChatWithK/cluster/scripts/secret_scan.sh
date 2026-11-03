#!/usr/bin/env bash
set -euo pipefail

patterns='(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|api[_-]?key[[:space:]]*[:=][[:space:]]*["'"'][^"'"']{12,}["'"']|bearer[[:space:]]+[A-Za-z0-9._-]{20,})'
if git grep -I -n -E "${patterns}" -- . ':!cluster/scripts/secret_scan.sh' ':!.env.example'; then
  echo "Potential committed secret detected." >&2
  exit 1
fi
echo "Secret scan passed."
