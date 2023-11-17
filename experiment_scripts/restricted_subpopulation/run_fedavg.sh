tmux new-session -d -s "session_fedavg"
tmux send-keys -t "session_fedavg" "mamba activate afl-bench" C-m
tmux send-keys -t "session_fedavg" "python afl_bench/experiments/fedavg.py -d cifar10 -dd restricted_subpopulation --subpopulation-size 5 --subpopulation-labels 0,1,2,3 -wff -bs 8 --client-info u1.0/2.0[5],u3.0/4.0[5] --num-aggregations 4000 --client-num-steps 5 -clr 0.001" C-m
tmux send-keys -t "session_fedavg" "exit" C-m
