
Autoregressive transformers can only learn their objective approximately, using


half of the tokens are the "next token", the other is bandwidth to stack information of the past

with this recursivity, it has a long term memory.

divide the dataset in 16 paralell batches

entropy of 0.46 is the classifier that learns to output 1 for x=10 and otherwise gives proba 1/2

## Goal
Will require more traning but at inference it should work almost as well as a traditional transformers, and better on long context tasks

## Architecture

half (right) of the input is the present X of the series, the other half are the previous embeddings


## Ideas
could work either in logits space
or directly in embedding (current)