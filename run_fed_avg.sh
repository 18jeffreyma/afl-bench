echo "Running FedAvg vs ExpWeight experiment with clients"
tmux new-session -d -s "session_fedavg"
tmux send-keys -t "session_fedavg" "mamba activate afl-bench" C-m
tmux send-keys -t "session_fedavg" "python afl_bench/experiments/fedavg.py -d cifar10 -dd randomly_remove --num-remove 7 -wff -bs 10 --client-info g0.0/4.0[10],g3.0/4.0[10],g6.0/4.0[10],g9.0/4.0[10] --num-aggregations 1000 --client-num-steps 10 -clr 0.001" C-m
tmux send-keys -t "session_fedavg" "exit" C-m


