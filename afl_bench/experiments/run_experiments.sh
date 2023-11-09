for i in {1..5}; do
  echo "Running experiment with $i clients"
  tmux new-session -d -s "session$i"
  tmux send-keys -t "session$i" "mamba activate afl-bench" C-m
  tmux send-keys -t "session$i" "python afl_bench/experiments/fedavg_cifar10.py -dd randomly_remove --num-remove $i -wff -bs 10 --num-clients 10 --num-aggregations 500" C-m
done