#!/bin/sh
# Installiert die neuen Tabellen-Datensätze data/train und data/val.
# Definition: challenge/PARKING.md — 2026-11-21 "Mehr Tabellen-Datensätze (train/val/…) definieren".
# Usage: ./scripts/gen_new_datasets.sh   (erzeugt data/train mit 5000, data/val mit 200 Tabellen)
set -e
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then . .venv/bin/activate; PY=python
elif [ -f venv/bin/activate ]; then . venv/bin/activate; PY=python
else PY=python3; fi
"$PY" generate_dataset.py --out data/train --n 5000 --seed 1
"$PY" generate_dataset.py --out data/val   --n 200  --seed 2
echo "Fertig: $(ls data/train | wc -l) Tabellen in data/train, $(ls data/val | wc -l) Tabellen in data/val"
