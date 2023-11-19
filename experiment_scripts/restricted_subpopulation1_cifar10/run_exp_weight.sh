tmux new-session -d -s "session_exp_weight"
tmux send-keys -t "session_exp_weight" "mamba activate afl-bench" C-m
tmux send-keys -t "session_exp_weight" "python afl_bench/experiments/exp_weighting.py --exp-weighting 0.5 -d cifar10 -dd restricted_subpopulation --subpopulation-size 5 --subpopulation-labels 0,1,2,3 -wff -bs 5 --client-info u1.0/2.0[10],u8.0/12.0[5] --num-aggregations 4000 --client-num-steps 1 -clr 0.01" C-m
tmux send-keys -t "session_exp_weight" "exit" C-m
