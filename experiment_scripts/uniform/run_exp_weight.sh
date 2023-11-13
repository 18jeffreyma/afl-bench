tmux new-session -d -s "session_exp_weight"
tmux send-keys -t "session_exp_weight" "mamba activate afl-bench" C-m
tmux send-keys -t "session_exp_weight" "python afl_bench/experiments/exp_weighting.py -d cifar10 -dd randomly_remove --num-remove 7 -wff -bs 10 --client-info u0.0/2.0[10],u2.0/4.0[10],u4.0/8.0[10],u8.0/12.0[10] --num-aggregations 2000 --client-num-steps 10 -clr 0.001" C-m
tmux send-keys -t "session_exp_weight" "exit" C-m
