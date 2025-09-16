#!/bin/bash
# Monitor system resources during experiment

LOG_FILE="logs/resource_usage.log"
mkdir -p logs

echo "Starting resource monitoring... (Ctrl+C to stop)"
echo "Log file: $LOG_FILE"

# Create header
echo "timestamp,cpu_percent,memory_percent,gpu_util,gpu_memory" > "$LOG_FILE"

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # CPU and Memory
    CPU_MEM=$(python3 -c "
import psutil
cpu = psutil.cpu_percent(interval=1)
mem = psutil.virtual_memory().percent
print(f'{cpu},{mem}')
")
    
    # GPU utilization if available
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
        GPU_UTIL=$(echo "$GPU_INFO" | cut -d',' -f1 | tr -d ' ')
        GPU_MEM=$(echo "$GPU_INFO" | cut -d',' -f2 | tr -d ' ')
    else
        GPU_UTIL="N/A"
        GPU_MEM="N/A"
    fi
    
    echo "$TIMESTAMP,$CPU_MEM,$GPU_UTIL,$GPU_MEM" >> "$LOG_FILE"
    
    # Display current status
    echo -ne "\r[$TIMESTAMP] CPU/MEM: $CPU_MEM | GPU: ${GPU_UTIL}% (${GPU_MEM} MiB)"
    
    sleep 30
done
