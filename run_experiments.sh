for i in 0; do
  echo "Running experiment with $i clients"
  tmux new-session -d -s "session$i"
  tmux send-keys -t "session$i" "mamba activate afl-bench" C-m
  tmux send-keys -t "session$i" "python afl_bench/experiments/fedavg.py -d cifar10 -dd randomly_remove --num-remove 5 -wff -bs 10 --client-info i0.0[20],g4.0/2.0[20] --num-aggregations 500 --client-num-steps 10 -clr 0.001" C-m
  tmux send-keys -t "session$i" "exit" C-m
done
