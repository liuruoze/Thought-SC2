# !/bin/bash

ITER=1
echo 'ITER is set to '${ITER}
python extract_sc_for_dream_1.py
python vae_train_sc_dream_1.py
python series_sc_dream_1.py
python rnn_train_sc_dream_1.py
python train_in_dream_1.py
#python eval_mini_srcgame_dream_${ITER}.py

echo 'All completed'


python vae_train_sc_dream_2.py
python series_sc_dream_2.py
python rnn_train_sc_dream_2.py
python train_in_dream_2.py

