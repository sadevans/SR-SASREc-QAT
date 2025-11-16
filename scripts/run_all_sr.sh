#!/bin/bash
# Run all SR experiments: FP32 + QAT + PTQ
set -e

echo "🚀 Training FP32 baseline..."
python -m experiments.sr.train --config configs/espcn_fp32.yaml

echo "🚀 Training QAT variants..."
python -m sr_experiments.train --config configs/quant_lsq.yaml
python -m sr_experiments.train --config configs/quant_pact.yaml
python -m sr_experiments.train --config configs/quant_apot.yaml
python -m sr_experiments.train --config configs/quant_qdrop.yaml

echo "🚀 Applying AdaRound PTQ (requires FP32 checkpoint)..."
python -m sr_experiments.train --config configs/quant_adaround.yaml

echo "🔍 Evaluating on Set5 (CPU)..."
for config in configs/espcn_fp32.yaml configs/quant_*.yaml; do
    name=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['experiment']['name'])")
    ckpt="runs/$name/best.ckpt"
    if [ -f "$ckpt" ]; then
        python -m sr_experiments.evaluate \
            --config "$config" \
            --checkpoint "$ckpt" \
            --split Set5
    else
        echo "⚠️  Checkpoint not found: $ckpt"
    fi
done

echo "✅ All experiments completed. Results in ./results/"