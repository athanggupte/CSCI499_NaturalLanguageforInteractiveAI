# Report and Analysis

## Models
### 1. LSTM Encoder-Decoder

This model employs simple RNNs with LSTM cells in both the encoder and decoder.

### 2. LSTM Encoder-Decoder with Attention

This model also uses RNNs with LSTM cells in both encoder and decoder, with added attention network that attends over the encoder outputs for predicting the next decoding step.
The attention is implemented to affect the output of the decoder and not the hidden states of the decoder itself.

3 variations of this model have been implemented and tested on the given dataset.

1. Vanilla attention
    - This is the base attention mechanism - global soft single-head attention.
    - `num_heads=1` and `attn_stride=0`
2. Multihead attention
    - This model uses multiple attention heads which independently attend upon the encoder outputs and are then added and normalized.
    - `num_heads>=2`
    - The multiple heads are expected to learn distinct patterns and features and thus provide a better understanding of the language to the model as a whole.
3. Multihead Local attention
    - This model has multiple attention heads and attends over a limited area governed by the _stride_ and _window_
    - `attn_stride>=1` and `attn_window`
    - Local attention is expected to limit the model's FOV (field of view) to a subset of the encoded inputs which may help to boost its "attention". Given that the input sequences are fairly long ~300+, the local attention should help the model focus on parts of the sentence that are more related to the current step of (action, target).


### 3. Transformer Encoder LSTM Decoder

This model has a Transformer encoder and an LSTM RNN with global soft attention over the encoder outputs.


## Metrics
### 1. Token-wise Accuracy
This checks the accuracy per token in the output sequence, i.e. every action token prediction is compared with its corresponding action token label and similarly for targets.

Pros:
1. Simple to implement and easy to understand.
2. Similar to regular accuracy in classification tasks.

Cons:
1. Does not penalize wrong pairing of action and target.
2. Does not take into account the ease of predicting actions vs targets. (Easy to predict correct from 8 actions vs 80 targets).

### 2. Pair-wise Accuracy
This checks the accuracy per pair of (action, target) tokens in the output sequence, i.e. a prediction is considered correct only if both the action and target are correct.

Pros:
1. Correctly penalizes pairs with mismatch between either action or target.
2. Simple to implement and easy to understand.

Cons:
1. Does not penalize if previous steps were wrong and only some step in between was correct. (Only predicting STOP correctly also increases this metric).
2. Does not take into account the ease of predicting actions vs targets.

### 3. Prefix-match Accuracy
This checks the accuracy of the sequence of (action, target) pairs from the beginning until the first mismatched prediction, i.e. it counts how many steps were predicted correctly before the first mistake.

Pros:
1. Correctly identifies and penalizes predictions that make a mistake early on in the sequence.

Cons:
1. Does not take into account the ease of predicting actions vs targets.
2. Cannot check if any rehabilitation or error-correction step was taken. (This is not a problem in this situation anyways since we're doing supervised learning and thus error-correction is not possible to be learned.)

### 4. Exact-match Accuracy
This is the most strict metric and awards accuracy points to only those predictions that predict the entire sequence of (action, target) pairs correctly from START to STOP.

Pros:
1. Correctly penalizes any mistake in the entire sequence.
2. Easy to understand and not complicated to implement.

Cons:
1. Does not award any partial credit for correctly predicting subset of steps.
2. Difficult to track progress of learning using this metric alone as it remains 0 for a lot of early epochs.



## Experiments and Results

All the tests were conducted with `21` epochs of training and with `1000` vocab size.

During training 50% of the samples in each mini-batch were randomly chosen to be trained using teacher-forcing and the remaining with student-forcing.
Initial attempts of training with only teacher-forcing resulted in extremely low validation accuracies with validation loss increasing each epoch in certain instances. Teaching with only student-forcing provided better results but soon capped at a fairly low accuracy score.
The ratio of teacher-forcing vs student-forcing is also another important hyperparameter that needs to be chosen. I suspect starting off with a higher ratio of teacher-forcing and then gradually increasing the ratio of student-forcing over time might provide faster learning efficiency along with good validation accuracies.

|model               |params            |exact|prefix|pairwise|token|
|--------------------|------------------|-----|------|--------|-----|
|lstm                |hidden_dim: 100   |0.0482|0.2062|0.5405|0.7195|
|                    |embedding_dim: 100|||||
|--                  |--                |||||
|attn                |hidden_dim: 100   |0.0000|0.1370|0.2227|0.4317|
|                    |embedding_dim: 100|||||
|                    |embedding_dim: 100|||||
|--                  |--                |||||
|attn-multihead      |hidden_dim: 100   |0.0000|0.1352|0.1989|0.4207|
|                    |embedding_dim: 100|||||
|                    |num_heads: 4      |||||
|--                  |--                |||||
|attn-multihead-local|hidden_dim: 100   |0.0000|0.2018|0.3950|0.5475|
|                    |embedding_dim: 100|||||
|                    |num_heads: 4      |||||
|                    |attn_stride: 15   |||||
|                    |attn_window: 25   |||||
|--                  |--                |||||
|trfm                |hidden_dim: 100   |0.0000|0.1382|0.2381|0.4407|
|                    |embedding_dim: 100|||||
|                    |num_heads: 4      |||||
|                    |num_layers: 4     |||||


Check the [saved_logs](saved_logs) directory for all the plots and logs for the above data.

### Observations and Inferences
1. We can see that in the limited training time and data, the vanilla LSTM network get the best performance in almost all the metrics. It is the only model that gets non-zero exact match score. It can be reasoned that since the model has lesser parameters, it trains faster as the gradients affect the lower (earlier) layers more efficiently.

2. It is also worth noticing that local soft attention works much better than global soft attention in this dataset and with the chosen hyperparameters. Since the input sequence lengths are very long (maximum is 375 tokens), I presume global attention fails to concentrate weights on the important parts thus assigning fat-tail-like weights across the encoded sequence.

3. I think the transformer model is bottlenecked by the global soft attending LSTM decoder. I suspect that using a transformer decoder with KQV attention may provide better results. Alas, the training times for that model were incompatible with the assignment deadline.

I think it is worth noting that 21 epochs may not be the ideal grounds for comparison of these models in terms of performance, since the larger models have significantly more number of parameters and thus are not as fast in updating the weights as the smaller models (due to gradient diminishing), however with more epochs they may outperform the simple LSTM.
