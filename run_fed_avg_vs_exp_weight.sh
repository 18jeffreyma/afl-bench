# echo "Running FedAvg vs ExpWeight experiment with clients"
# tmux new-session -d -s "session_fedavg"
# tmux send-keys -t "session_fedavg" "mamba activate afl-bench" C-m
# tmux send-keys -t "session_fedavg" "python afl_bench/experiments/fedavg.py -d cifar10 -dd randomly_remove --num-remove 5 -wff -bs 10 --client-info i0.0[20],g4.0/2.0[20] --num-aggregations 500 --client-num-steps 10 -clr 0.001" C-m
# tmux send-keys -t "session_fedavg" "exit" C-m

tmux new-session -d -s "session_exp_weight"
tmux send-keys -t "session_exp_weight" "mamba activate afl-bench" C-m
tmux send-keys -t "session_exp_weight" "python afl_bench/experiments/exp_weighting.py -d cifar10 -dd randomly_remove --num-remove 5 -wff -bs 10 --client-info i0.0[20],g4.0/2.0[20] --num-aggregations 500 --client-num-steps 10 -clr 0.001" C-m
tmux send-keys -t "session_exp_weight" "exit" C-m
