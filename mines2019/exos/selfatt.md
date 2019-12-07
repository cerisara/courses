
# Self-attention

- Create 3 random pytorch tensors, with dimension n: they represent your sequence input (say, words embeddings in a sentence, image embeddings in a video, sensor values in a time series...)

We're going to compute the attention between every pair of items in the sequence: this is self-attention.
To do this, we'll use 3 matrices that respectively transform a given input tensor into a *query* tensor, a *key* tensor, and a *value* tensor.

- Create 3 random matrices for queries, keys and values
- Compute, in one operation, all the key tensors from all input tensors. You can do the same for values, and then for queries.

Let's compute the attention for each input vector X:

- Compute the dot-product between the *queries* Q and all *keys* K1,K2,K3

You should get 3x3 attention scores (scalar), which have to be normalized:

- Normalize these attention scores with a softmax

These attention scores indicate which *inputs* the model is looking at.
But the *output* of the attention is the weighted sum of the corresponding *values*. So,

- Compute the sum of the input *value* tensors, weighted by their corresponding attention score

You thus obtain one vector per input, i.e., a new sequence of vectors. 
In the *Transformer* model, it is the input of a next layer of self-attention.

- Compress this variable-length final sequence into a single fixed-size vector with max-pooling
- Add a final classification layer to predict 2 classes and put all this inside a pytorch nn.Module model

## Limitations

- Generate random sequences with $n=1$:
    - class A: the sequence is composed of observations uniformely sampled between 0 and 1
    - class B: the sequence is composed of observations uniformely sampled between 1 and 2
- Train your self-attentive classifier
- Analyze
- Generate random sequences with $n=1$:
    - class A: the first half of the sequence is composed of observations uniformely sampled between 0 and 1, and the second half between 1 and 2
    - class B: the first half between 1 and 2, and the second half between 0 and 1
- Train your self-attentive classifier
- Analyze

## Refs

http://jalammar.github.io/illustrated-transformer/

