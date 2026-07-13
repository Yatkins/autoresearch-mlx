#!/bin/bash
# run.sh — start an autoresearch session
# Usage: ./run.sh
# Launches Claude Code pointed at program.md.
# All API keys are loaded from .env.local by evaluate.py at runtime — no export needed here.

set -e

echo "=== Invoice OCR Autoresearch ==="
echo ""

# Verify environment
if [ ! -f ".env.local" ]; then
    echo "ERROR: .env.local not found. Create it with your API keys."
    exit 1
fi

if [ ! -d "Training Invoices" ]; then
    echo "ERROR: 'Training Invoices' directory not found."
    exit 1
fi

if [ ! -f "InvoiceFormat.json" ]; then
    echo "ERROR: InvoiceFormat.json not found."
    exit 1
fi

# Verify git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repo..."
    git init
    git add score.py evaluate.py program.md InvoiceFormat.json run.sh
    git commit -m "initial setup"
fi

# Show current best if results exist
if [ -f "results.tsv" ]; then
    BEST=$(sort -t$'\t' -k1 -rn results.tsv | head -1 | cut -f1)
    COUNT=$(wc -l < results.tsv)
    echo "Current best score: $BEST  (from $COUNT experiments)"
else
    echo "No results yet — this will be the first run."
    touch results.tsv
fi

echo ""
echo "Starting Claude Code. It will read program.md and begin the loop."
echo "Stop with Ctrl+C when you want to pause."
echo ""

# Launch Claude Code in the repo directory
claude