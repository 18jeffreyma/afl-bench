tmux new-session -d -s "session_reverse_exp_weight"
tmux send-keys -t "session_reverse_exp_weight" "mamba activate afl-bench" C-m
tmux send-keys -t "session_reverse_exp_weight" "python afl_bench/experiments/reverse_exp_weighting.py -d cifar10 -dd randomly_remove --num-remove 7 -wff -bs 10 --client-info g0.0/4.0[10],g3.0/4.0[10],g6.0/4.0[10],g9.0/4.0[10] --num-aggregations 2000 --client-num-steps 10 -clr 0.001" C-m
tmux send-keys -t "session_reverse_exp_weight" "exit" C-m
