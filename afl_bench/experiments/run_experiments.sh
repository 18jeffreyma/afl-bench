for i in {1..5}; do
  echo "Running experiment with $i clients"
  tmux new-session -d -s "session$i"
  tmux send-keys -t "session$i" "mamba activate afl-bench" C-m
  tmux send-keys -t "session$i" "python afl_bench/experiments/fedavg_cifar10.py -d cifar10 -dd randomly_remove --num-remove $i -wff -bs 5 --client-info i0.0[20] --num-aggregations 500 --client-num-steps 10 -clr 0.001" C-m
done