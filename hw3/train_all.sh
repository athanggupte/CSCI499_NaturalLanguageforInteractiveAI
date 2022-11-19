#!/bin/sh
# lstm
python3 train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=models --num_epochs=21 --batch_size=512 --model_type=lstm
# attention (vanilla)
python3 train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=models --num_epochs=21 --batch_size=512 --model_type=attn
# attention (multihead)
python3 train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=models --num_epochs=21 --batch_size=512 --model_type=attn --num_attn_heads=4 --output_plot_fn=multihead
# attention (multihead, local)
python3 train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=models --num_epochs=21 --batch_size=512 --model_type=attn --num_attn_heads=4 --attn_stride=15 --output_plot_fn=multihead-local
# transformer
python3 train.py --in_data_fn=lang_to_sem_data.json --model_output_dir=models --num_epochs=26 --batch_size=128 --model_type=trfm --num_attn_heads=4
