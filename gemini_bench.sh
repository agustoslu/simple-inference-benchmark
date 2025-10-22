#!/bin/bash

# Before launching, please make sure to install GNU Parallel
# sudo apt-get update && sudo apt-get install parallel
# Please make sure you change the path to your virtual environment accordingly

VENV_PATH="/home/tanalp/toxicainment/toxicainment_venv" 
VENV_PYTHON="${VENV_PATH}/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python executable not found at $VENV_PYTHON"
    exit 1
fi

MODALITIES=("only_video" "transcribed" "muted" "muted_no_transcript")
STOCHASTIC_RUNS=5
MAX_PARALLEL_JOBS=2
SESSION_NAME="gemini_bench"
LOG_DIR="benchmark_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Log files will be saved in: $LOG_DIR"

tmux kill-session -t "$SESSION_NAME" 2>/dev/null
tmux new-session -d -s "$SESSION_NAME" -n "batch_runner"
echo "Created tmux session '$SESSION_NAME'. Launching GNU Parallel inside..."

COMMAND="
for modality in ${MODALITIES[@]}; do
    for i in \$(seq 1 $STOCHASTIC_RUNS); do
        RUN_ID=\$(uuidgen)
        LOG_FILE=\"${LOG_DIR}/\${modality}_run_\${i}.log\"

        echo \"set -e; $VENV_PYTHON run_benchmark.py --model_id='gemini-2.5-flash' --n_examples=300 --input_strategy \${modality} --run_id \${RUN_ID} > \${LOG_FILE} 2>&1\"
    done
done | parallel --bar -j $MAX_PARALLEL_JOBS --retries 3 --joblog ${LOG_DIR}/batch.log
"

tmux send-keys -t "${SESSION_NAME}:batch_runner" "$COMMAND" C-m

echo ""
echo "All experiments have been launched inside tmux session '$SESSION_NAME'."
echo "----------------------------------------------------"
echo "To attach and see the progress bar, use: tmux attach -t $SESSION_NAME"
echo "To detach safely (and leave it running), press: Ctrl+b + : + detach"
echo "To monitor all logs, use: tail -f ${LOG_DIR}/*.log"
echo "To see the master job log, use: cat ${LOG_DIR}/batch.log"
echo "----------------------------------------------------"