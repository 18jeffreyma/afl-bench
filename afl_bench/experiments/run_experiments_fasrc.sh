for i in {0..5}; do
  echo "Running experiment with $i clients"
  tmux new-session -d -s "session$i"
  tmux send-keys -t "session$i" "afl_bench" C-m
  tmux send-keys -t "session$i" "python afl_bench/experiments/fedavg_cifar10.py -d cifar10 -dd randomly_remove --num-remove $i -wff -bs 10 --client-info i0.0[50] --num-aggregations 500 --client-num-steps 10 -clr 0.001" C-m
done