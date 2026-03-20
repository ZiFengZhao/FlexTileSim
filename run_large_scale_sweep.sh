#!/bin/bash

#WORKLOADS=("alexnet" "resnet18" "vgg16" "bert_base" "vit_large")
WORKLOADS=("bert_base" "vit_large")
CORE_NUMS=(32)
#CORE_NUMS=(4)
CONFIG_TEMPLATE="./config/template_cfg.txt"
MODEL_DEFINE_DIR="./transform_compiler/model_define"
SIM_EXE="./build/npu_sim"
LOG_DIR="./log"
CFG_DIR="./config"

TIMEOUT_DURATION="4h"  # "2h"、"30m"、"60s"
TIMEOUT_CMD="timeout"  

TIMESTAMP=$(date +"%Y%m%d_%H%M")
RESULT_FILE="sweep_results_${TIMESTAMP}.txt"
ERROR_LOG="${LOG_DIR}/error_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

echo "Simulation sweep started at $(date)" > "$RESULT_FILE"
echo "Error log: $ERROR_LOG" >> "$RESULT_FILE"
echo "Timeout duration: $TIMEOUT_DURATION" >> "$RESULT_FILE"
echo "" >> "$RESULT_FILE"

# noc_mode: 0=detail, 1=analytic
# ddr_mode: 0=detail, 1=analytic
NOC_MODES=(1 0)
DDR_MODES=(1 0)

TOTAL_STEPS=$((${#CORE_NUMS[@]} * ${#WORKLOADS[@]} * ${#NOC_MODES[@]} * ${#DDR_MODES[@]}))
CURRENT_STEP=0

if ! command -v $TIMEOUT_CMD &> /dev/null; then
    echo "Warning: 'timeout' command not found. Timeout functionality will be disabled."
    echo "Please install coreutils if you need timeout feature."
    HAS_TIMEOUT=0
else
    HAS_TIMEOUT=1
fi

for CORE in "${CORE_NUMS[@]}"; do
    for WORKLOAD in "${WORKLOADS[@]}"; do
        echo "[$CORE cores][$WORKLOAD] Generating assembly..."
        python3 "${MODEL_DEFINE_DIR}/frontend_exp.py" \
            -b \
            --model "$WORKLOAD" \
            --csv_path "${MODEL_DEFINE_DIR}/${WORKLOAD}.csv" \
            --tile_num "$CORE"

        if [ $? -ne 0 ]; then
            echo "Error: Failed to generate binary for $WORKLOAD with $CORE cores" | tee -a "$ERROR_LOG"
            continue
        fi

        BIN_FILE="${WORKLOAD}_${CORE}tile_inst.bin"

        for NOC_MODE in "${NOC_MODES[@]}"; do
            for DDR_MODE in "${DDR_MODES[@]}"; do
                ((CURRENT_STEP++))
                echo "Running simulation [$CURRENT_STEP/$TOTAL_STEPS]: $WORKLOAD, cores=$CORE, noc_mode=$NOC_MODE, ddr_mode=$DDR_MODE"

                CFG_FILE="${CFG_DIR}/${WORKLOAD}_${CORE}tile_noc${NOC_MODE}_ddr${DDR_MODE}.txt"
                cp "$CONFIG_TEMPLATE" "$CFG_FILE"

                sed -i "s/^core_num = .*/core_num = $CORE/" "$CFG_FILE"
                sed -i "s/^noc_mode = .*/noc_mode = $NOC_MODE/" "$CFG_FILE"
                sed -i "s/^ddr_mode = .*/ddr_mode = $DDR_MODE/" "$CFG_FILE"
                sed -i "s|^inst_file = .*|inst_file = ${BIN_FILE}|" "$CFG_FILE"
                
                TEMP_OUTPUT=$(mktemp)
                TEMP_ERROR=$(mktemp)
                
                START_TIME=$(date +%s)
                
                if [ $HAS_TIMEOUT -eq 1 ]; then
                    $TIMEOUT_CMD $TIMEOUT_DURATION "$SIM_EXE" "$CFG_FILE" > "$TEMP_OUTPUT" 2> "$TEMP_ERROR"
                    EXIT_CODE=$?
                else
                    "$SIM_EXE" "$CFG_FILE" > "$TEMP_OUTPUT" 2> "$TEMP_ERROR"
                    EXIT_CODE=$?
                fi
                
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))
                
                if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 137 ]; then
                    WARNING_MSG="WARNING: Simulation TIMEOUT after ${TIMEOUT_DURATION} for $WORKLOAD cores=$CORE noc=$NOC_MODE ddr=$DDR_MODE"
                    echo "$WARNING_MSG" | tee -a "$ERROR_LOG"
                    
                    echo "========== [$WORKLOAD cores=$CORE noc=$NOC_MODE ddr=$DDR_MODE] ==========" >> "$RESULT_FILE"
                    echo "STATUS: TIMEOUT after ${TIMEOUT_DURATION}" >> "$RESULT_FILE"
                    echo "Duration: ${DURATION} seconds" >> "$RESULT_FILE"
                    echo "Warning: Simulation did not complete within the time limit" >> "$RESULT_FILE"
                    
                    if [ -s "$TEMP_ERROR" ]; then
                        echo "Error output:" >> "$RESULT_FILE"
                        cat "$TEMP_ERROR" >> "$RESULT_FILE"
                    fi
                    
                    echo "" >> "$RESULT_FILE"
                    
                elif [ $EXIT_CODE -ne 0 ]; then
                    ERROR_MSG="ERROR: Simulation failed with exit code $EXIT_CODE for $WORKLOAD cores=$CORE noc=$NOC_MODE ddr=$DDR_MODE"
                    echo "$ERROR_MSG" | tee -a "$ERROR_LOG"
                    
                    echo "========== [$WORKLOAD cores=$CORE noc=$NOC_MODE ddr=$DDR_MODE] ==========" >> "$RESULT_FILE"
                    echo "STATUS: FAILED (exit code: $EXIT_CODE)" >> "$RESULT_FILE"
                    echo "Duration: ${DURATION} seconds" >> "$RESULT_FILE"
                    
                    if [ -s "$TEMP_ERROR" ]; then
                        echo "Error output:" >> "$RESULT_FILE"
                        cat "$TEMP_ERROR" >> "$RESULT_FILE"
                    fi
                    
                    echo "" >> "$RESULT_FILE"
                    
                else
                    echo "Simulation completed in ${DURATION} seconds"
                    
                    STAT_START=$(grep -n "Simulation Statistics:" "$TEMP_OUTPUT" | cut -d: -f1)
                    if [ -n "$STAT_START" ]; then
                        STAT_CONTENT=$(tail -n +$STAT_START "$TEMP_OUTPUT")
                        echo "========== [$WORKLOAD cores=$CORE noc=$NOC_MODE ddr=$DDR_MODE] ==========" >> "$RESULT_FILE"
                        echo "Duration: ${DURATION} seconds" >> "$RESULT_FILE"
                        echo "$STAT_CONTENT" >> "$RESULT_FILE"
                        echo "" >> "$RESULT_FILE"
                    else
                        WARNING_MSG="Warning: Simulation Statistics not found for $WORKLOAD cores=$CORE noc=$NOC_MODE ddr=$DDR_MODE"
                        echo "$WARNING_MSG" | tee -a "$ERROR_LOG"
                        
                        echo "========== [$WORKLOAD cores=$CORE noc=$NOC_MODE ddr=$DDR_MODE] ==========" >> "$RESULT_FILE"
                        echo "STATUS: NO STATISTICS FOUND" >> "$RESULT_FILE"
                        echo "Duration: ${DURATION} seconds" >> "$RESULT_FILE"
                        echo "Simulation output:" >> "$RESULT_FILE"
                        cat "$TEMP_OUTPUT" >> "$RESULT_FILE"
                        echo "" >> "$RESULT_FILE"
                    fi
                fi
                
                rm -f "$TEMP_OUTPUT" "$TEMP_ERROR"
                rm -f "$CFG_FILE"
            done
        done
        rm -f "$BIN_FILE"
    done
done

echo "" >> "$RESULT_FILE"
echo "=========================================" >> "$RESULT_FILE"
echo "Simulation Sweep Summary" >> "$RESULT_FILE"
echo "=========================================" >> "$RESULT_FILE"
echo "Total experiments: $TOTAL_STEPS" >> "$RESULT_FILE"
echo "Completed at: $(date)" >> "$RESULT_FILE"
if [ -f "$ERROR_LOG" ]; then
    ERROR_COUNT=$(grep -c "WARNING\|ERROR" "$ERROR_LOG" 2>/dev/null || echo "0")
    echo "Warnings/Errors: $ERROR_COUNT (see $ERROR_LOG for details)" >> "$RESULT_FILE"
fi
echo "Results saved to $RESULT_FILE"
echo "Error log saved to $ERROR_LOG"

echo "Simulation sweep completed at $(date)" >> "$RESULT_FILE"
echo "Results saved to $RESULT_FILE"
