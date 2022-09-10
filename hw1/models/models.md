### lstm
```sh
time python3 train.py --in_data_fn=lang_to_sem_data.json --emb_dim=200 --lstm_dim=200 --learning_rate=0.1 --num_epochs=20 --vocab_size=1000 --batch_size=512 --model_type=lstm --no_len_limit
```

### lstm + prev
```sh
time python3 train.py --in_data_fn=lang_to_sem_data.json --emb_dim=200 --lstm_dim=200 --learning_rate=0.1 --num_epochs=20 --vocab_size=1000 --batch_size=512 --model_type=lstm --no_len_limit --context=prev
```

### lstm + next
```sh
time python3 train.py --in_data_fn=lang_to_sem_data.json --emb_dim=200 --lstm_dim=200 --learning_rate=0.1 --num_epochs=20 --vocab_size=1000 --batch_size=512 --model_type=lstm --no_len_limit --context=next
```

### attn
```sh
time python3 train.py --in_data_fn=lang_to_sem_data.json --emb_dim=200 --lstm_dim=200 --learning_rate=0.1 --num_epochs=20 --vocab_size=1000 --batch_size=512 --model_type=attn
```

### act-tar
```sh
time python3 train.py --in_data_fn=lang_to_sem_data.json --emb_dim=200 --lstm_dim=200 --learning_rate=0.1 --num_epochs=20 --vocab_size=1000 --batch_size=512 --model_type=act-tar --no_len_limit
```

### tar-act
```sh
time python3 train.py --in_data_fn=lang_to_sem_data.json --emb_dim=200 --lstm_dim=200 --learning_rate=0.1 --num_epochs=20 --vocab_size=1000 --batch_size=512 --model_type=act-tar --no_len_limit
```
