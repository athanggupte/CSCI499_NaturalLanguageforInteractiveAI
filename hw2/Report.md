# Homework-2 Report
Author: Athang Gupte
Date: Oct-14-2022

## Implementation

### Continuous Bag-of-Words or CBOW

#### Data:
- Input is a 2D vector of size `(data_size, context_len * 2)` where each vector has indices of `context_len` words on the left and `context_len` words on the right of the observed word.
- Output is a 2D vector of size `(data_size, 1)` where each vector has a single index corresponding to the word under observation.

#### Model:
- Embedding
    - `num_embeddings`: 3000
    - `embedding_dim`: 300 or 180
    - output is the set of embedding vectors for the given input tokens
- Mean
- Linear
    - `in_features`: 300 or 180
    - `out_features`: 3000
    - output is a vector with elements denoting probability density of each token being the most probable token given the context tokens
    
(In PyTorch outputs aren't explicitly converted to probability densities using softmax as this is an implicit step in `CrossEntropyLoss`)

#### Process:
1. All the sentences are read and converted into vectors of word indices.
2. Each sentence is padded by `context_len` `<pad>` tokens on both left and right.
3. A sliding window of length `2 * context_len + 1` slides over each sentence with a stride of 1 token.
4. At each instance, the center of the context window is stored as the output and the remaining tokens are stored as the inputs.

#### Loss Function:
Cross-entropy loss is used to push the output logits to have most weight on the correct token.

### Continuous Skip-Gram

#### Data:
- Input is a 2D vector of size `(data_size, 1)` where each vector has a single index corresponding to the word under observation.
- Output is a 2D vector of size `(data_size, context_len * 2)` where each vector has the indices of `context_len` words on the left and `context_len` words on the right of the observed word.

#### Model:
- Embedding
    - `num_embeddings`: 3000
    - `embedding_dim`: 300 or 180
    - output is the embedding vector for the input token
- Linear
    - `in_features`: 300 or 180
    - `out_features`: 3000
    - output is a vector where each element is a logit denoting the probability of the token occuring in the context for the input context.

(in PyTorch outputs aren't explicitly converted to logits using sigmoid as this is an implicit step in `BCEWithLogitsLoss`)

#### Process:
Same as CBOW except the inputs and outputs are swapped.

#### Loss Function:
Binary cross-entropy loss is used. Here, each output logit is a binary classification problem - "Does this token occur in the context of the input token or not?"  
Binary cross-entropy requires the tensors in the form of multi-hot encoding. hot encoding with 1's in position of the words contained in the context.

#### Accuracy measurement:
__Intersection-over-Union or IoU:__  
For each training example, the set of unique tokens in the predicted output context $P$ and the actual context $L$ is created. Then intersection-over-union is given by -
$$ \frac{\| P \cap L \|}{\| P \cup L \|}$$


#### Hyperparameters

| Parameter | Value |  |
|--|--:|--|
| Context Length | 4 | Default value chosen by original authors |
| Embedding Dimension | 300 |  |
|  | 180 |  |

## Analysis of Downstream Tasks

The downstream tasks involve 2 types of relational analogies - Semantic and Syntactic.

Syntactic analogies include relations based mainly on grammatic syntax such as tense [run, ran, running], comparison degrees [good, better, best], singular-plural [boy, boys], adjective-adverb [great, greatly], etc.

Semantic analogies include relations that occur based on the meaning or context of use such as sex [king, queen], antonyms [same, other], cause-effect [raise, rise], membership [ship, fleet], etc.

The performance of the models on these analogies is measured by 3 different scores -
- Exact: the expected token is the closest token to the target position in the embedding space (1 if closest else 0)
- Mean Rank (MR): the rank of the expected token when ordered by increasing closeness from the target position
- Mean Reciprocal Rank (MRR): the reciprocal of the rank of the expected token when ordered by increasing closeness from the expected position

Exact is the strictest metric and penalizes even the slightest error.  
MRR gives an idea of a "score" of how close the embedding vector is to the desired position.  
MR gives the average of how many other tokens are erroneously placed closer to the target than the expected token.


### Runs:
1. __run-1:__ CBOW, 300-dim, context-width 4, 21 epochs
1. __run-2:__ Skip-Gram, 180-dim, context-width 4, 21 epochs
1. __run-3:__ CBOW, 300-dim, context-width 4, 21 epochs
1. __run-4:__ Skip-Gram, 180-dim, context-width 4, 21 epochs

Results and plots are in [logs/](logs/) directory.

