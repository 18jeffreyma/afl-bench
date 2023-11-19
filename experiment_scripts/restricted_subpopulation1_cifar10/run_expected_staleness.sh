tmux new-session -d -s "session_expected_staleness"
tmux send-keys -t "session_expected_staleness" "mamba activate afl-bench" C-m
tmux send-keys -t "session_expected_staleness" "python afl_bench/experiments/expected_staleness.py -d cifar10 -dd restricted_subpopulation --subpopulation-size 5 --subpopulation-labels 0,1,2,3 -wff -bs 5 --client-info u1.0/2.0[10],u8.0/12.0[5] --num-aggregations 16000 --client-num-steps 1 -clr 0.01" C-m
tmux send-keys -t "session_expected_staleness" "exit" C-m