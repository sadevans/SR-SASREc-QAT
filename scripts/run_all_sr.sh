#!/bin/bash
# Run all SR experiments: FP32 + QAT + PTQ
set -e

# echo "🚀 Training FP32 baseline..."
# python -m experiments.sr.train --config configs/sr/espcn_fp32.yaml

# echo "🚀 Applying AdaRound PTQ (requires FP32 checkpoint)..."
# python -m experiments.sr.train --config configs/sr/quant_adaround.yaml

# echo "🚀 Training QAT variants..."
# echo "  - LSQ..."
# python -m experiments.sr.train --config configs/sr/quant_lsq.yaml

# echo "  - QDrop..."
# python -m experiments.sr.train --config configs/sr/quant_qdrop.yaml

# echo "  - PACT..."
# python -m experiments.sr.train --config configs/sr/quant_pact.yaml

# echo "  - APoT..."
# python -m experiments.sr.train --config configs/sr/quant_apot.yaml

echo "🔍 Evaluating all models on Set14 (CPU)..."
for config in configs/sr/espcn_fp32.yaml configs/sr/quant_*.yaml; do
    if [ ! -f "$config" ]; then
        continue
    fi
    
    name=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['experiment']['name'])")
    ckpt="runs/$name/best.ckpt"
    if [ -f "$ckpt" ]; then
        echo "  Evaluating $name..."
        python -m experiments.sr.evaluate \
            --config "$config" \
            --checkpoint "$ckpt" \
            --split Set14
    else
        echo "⚠️  Checkpoint not found: $ckpt (skipping $name)"
    fi
done

echo "✅ All experiments completed. Results in ./results/"