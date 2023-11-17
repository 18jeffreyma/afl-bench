tmux new-session -d -s "session_reverse_exp_weight"
tmux send-keys -t "session_reverse_exp_weight" "mamba activate afl-bench" C-m
tmux send-keys -t "session_reverse_exp_weight" "python afl_bench/experiments/reverse_exp_weighting.py -d cifar10 -dd restricted_subpopulation --subpopulation-size 5 --subpopulation-labels 0,1,2,3 -wff -bs 8 --client-info u1.0/2.0[5],u3.0/4.0[5] --num-aggregations 4000 --client-num-steps 5 -clr 0.001" C-m
tmux send-keys -t "session_reverse_exp_weight" "exit" C-m
