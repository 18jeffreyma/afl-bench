tmux new-session -d -s "session_fedavg"
tmux send-keys -t "session_fedavg" "mamba activate afl-bench" C-m
tmux send-keys -t "session_fedavg" "python afl_bench/experiments/fedavg.py -d fashion_mnist -dd restricted_subpopulation --subpopulation-size 5 --subpopulation-labels 0,1,2,3 -wff -bs 5 --client-info u1.0/2.0[10],u8.0/12.0[5] --num-aggregations 2000 --client-num-steps 1 -clr 0.01" C-m
tmux send-keys -t "session_fedavg" "exit" C-m
