#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

apt-get update
apt-get install -y pandoc texlive-xetex

pandoc writeup/writeup.md -o Safety_Delta_Atlas_WRITEUP.pdf --pdf-engine=xelatex
pandoc writeup/exec_summary.md -o Safety_Delta_Atlas_EXEC_SUMMARY.pdf --pdf-engine=xelatex

echo "[OK] PDFs generated at repo root."
