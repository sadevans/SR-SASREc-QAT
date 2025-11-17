#!/bin/bash
# Run all SR experiments: FP32 + QAT + PTQ
set -e

echo "🚀 Training FP32 baseline..."
python -m experiments.sr.train --config configs/sr/espcn_fp32.yaml

echo "🚀 Applying AdaRound PTQ (requires FP32 checkpoint)..."
python -m experiments.sr.train --config configs/sr/quant_adaround.yaml

echo "🚀 Training QAT variants..."
python -m experiments.sr.train --config configs/sr/quant_lsq.yaml
python -m experiments.sr.train --config configs/sr/quant_pact.yaml
python -m experiments.sr.train --config configs/sr/quant_apot.yaml
python -m experiments.sr.train --config configs/sr/quant_qdrop.yaml

echo "🔍 Evaluating on Set14 (CPU)..."
for config in configs/sr/espcn_fp32.yaml configs/sr/quant_*.yaml; do
# for config in configs/sr/espcn_fp32.yaml configs/sr/quant_adaround.yaml; do

    name=$(python -c "import yaml; print(yaml.safe_load(open('$config'))['experiment']['name'])")
    ckpt="runs/$name/best.ckpt"
    if [ -f "$ckpt" ]; then
        python -m experiments.sr.evaluate \
            --config "$config" \
            --checkpoint "$ckpt" \
            --split Set14
    else
        echo "⚠️  Checkpoint not found: $ckpt"
    fi
done

echo "✅ All experiments completed. Results in ./results/"