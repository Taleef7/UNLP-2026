#!/bin/bash
# Monitor cp310 wheel build and verify when complete
LOG=/scratch/gilbreth/tamst01/unlp2026/logs/llama_sm60_cp310_wheel.log
WHEELS=/scratch/gilbreth/tamst01/unlp2026/kaggle_datasets/unlp2026-wheels

while true; do
    # Check if cp310 wheel appeared
    WHL=$(ls $WHEELS/llama_cpp_python-0.3.16-cp310-*.whl 2>/dev/null | head -1)
    if [ -n "$WHL" ]; then
        echo "=== cp310 wheel built: $WHL ==="
        ls -lh $WHL
        # Verify sm_60 target
        pip install --target /tmp/test_cp310 "$WHL" --no-deps -q 2>/dev/null
        RESULT=$(strings /tmp/test_cp310/llama_cpp/lib/libggml-cuda.so 2>/dev/null | grep "\.target sm_60" | head -1)
        if [ -n "$RESULT" ]; then
            echo "VERIFIED: $RESULT"
        else
            echo "WARNING: Could not verify sm_60 in wheel"
        fi
        rm -rf /tmp/test_cp310
        break
    fi
    # Check for errors
    if grep -q "ERROR\|error:" $LOG 2>/dev/null; then
        echo "Build error detected:"
        grep "ERROR\|error:" $LOG | tail -5
        break
    fi
    echo "$(date +%H:%M): Still building... ($(find /tmp/tmp*llama* -name '*.cu.o' 2>/dev/null | wc -l)/174 objects)"
    sleep 60
done
