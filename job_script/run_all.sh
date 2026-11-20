#!/bin/bash
# Run every script listed in names.csv of one analysis folder for the 5 held-out split seeds (0-4).
# Each run writes its results (Metrics_, History_, Sampled_, Pred_ files) and one log per seed
# to <analysis folder>/<run>/, the layout collect_metrics.py, ablation_summary.py and median_seed.py read.
#
# Usage, from the repository folder:
#   bash job_script/run_all.sh <analysis folder> [text]
#   e.g. bash job_script/run_all.sh CO2-CH4/cubtc/AF_Test
#        bash job_script/run_all.sh CO2-CH4/cubtc/AF_Test TAL      only the runs whose name contains TAL
# then:
#   python job_script/collect_metrics.py CO2-CH4/cubtc/AF_Test     (All_metrics.csv, Summary_metrics.csv)

if [ -z "$1" ] || [ ! -f "$1/names.csv" ]; then
    echo "Usage: bash job_script/run_all.sh <analysis folder with names.csv> [text]"
    exit 1
fi
folder=$(cd "$1" && pwd)

for run in $(grep -- "${2:-}" "$folder/names.csv"); do
    mkdir -p "$folder/$run"
    for seed in 0 1 2 3 4; do
        echo "$run seed $seed"
        if ! (cd "$folder/$run" && python "../$run.py" $seed > "log_seed$seed.txt" 2>&1); then
            echo "  failed, see $folder/$run/log_seed$seed.txt"
        fi
    done
done
