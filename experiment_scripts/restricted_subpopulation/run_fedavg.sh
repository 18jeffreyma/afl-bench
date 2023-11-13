tmux new-session -d -s "session_fedavg"
tmux send-keys -t "session_fedavg" "mamba activate afl-bench" C-m
tmux send-keys -t "session_fedavg" "python afl_bench/experiments/fedavg.py -d cifar10 -dd restricted_subpopulation --subpopulation-size 10 --subpopulation-labels 0,1,2,3 -wff -bs 10 --client-info u0.0/4.0[30],u8.0/12.0[10] --num-aggregations 4000 --client-num-steps 10 -clr 0.001" C-m
tmux send-keys -t "session_fedavg" "exit" C-m
