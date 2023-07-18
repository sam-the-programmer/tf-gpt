print("Importing dependencies")
import argparse as arg
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.metrics as metrics
import tensorflow.keras.optimizers as optim
import tensorflow.keras.activations as activ
import tensorflow.keras.backend as K
import tqdm
from numba import jit
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import (
    Add, Dense, Dropout, Embedding,
    Flatten, Input, LayerNormalization, MultiHeadAttention,
    Reshape,
)
from tensorflow.keras.utils import plot_model

argparser = arg.ArgumentParser()
argparser.add_argument("-d", "--data")

args = argparser.parse_args()

print("Loading data")
if not args.data:
  dataset = load_dataset("wikitext", "wikitext-103-v1")
  txt = "\n".join(dataset["train"]["text"])
  del dataset
else:
  with open(args.data) as file:
    txt = file.read()

print("Setting hyperparameters")
# Constants
LOOKBACK = 64 # context window
TRAIN_N_SAMPLES = 2000000 # number of tokens to train on
TRAIN_N_EPOCHS = 10 # epochs to train for
TRAIN_SIZE_BATCH = 128 # batch size

EMBEDDING_SIZE = 64 # embedding space

N_BLOCKS = 4 # number gpt decoder blocks
assert N_BLOCKS >= 1, "Must have at least 1 block!"

BASE_DROPOUT = 0.2 # dropout on block end
BASE_LINEAR_SIZE = 2048 # size of GeLU dense layer

ATTENTION_HEADS = 6 # number of multi-headed attention heads
ATTENTION_KEY_SIZE = 128 # key and query vector space
ATTENTION_VALUE_SIZE = 128 # value vector space
ATTENTION_DROPOUT = 0.2 # dropout during attention
ATTENTION_MASK = True # use a decoder look-ahead mask

tokens = "".join(sorted(set(txt)))
VOCAB_SIZE = len(tokens)

print("Encoding Dataset")
def encode(x):
    return np.fromiter(map(tokens.index, x), dtype=np.float32)

randoms = np.random.randint(0, len(txt) - LOOKBACK, size=(TRAIN_N_SAMPLES,)) # data point indexes
X = [encode(txt[i: i+LOOKBACK]) for i in tqdm.tqdm(randoms)]
y = [tokens.index(txt[i+LOOKBACK]) for i in tqdm.tqdm(randoms)]
X = np.array(X)
y = np.array(y)

# Models
K.clear_session()

print("Creating model subclasses")
class PositionEmbeddingFixedWeights(layers.Layer):
  def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
    super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
    position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)

    self.word_embedding_layer = Embedding(
        input_dim=vocab_size, output_dim=output_dim,
        input_length=sequence_length,
        trainable=True
    )
    self.position_embedding_layer = Embedding(
        input_dim=sequence_length, output_dim=output_dim,
        input_length=sequence_length,
        weights=[position_embedding_matrix],
        trainable=False
    )
  
  def get_position_encoding(self, seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P
  
  
  def call(self, inputs):
    position_indices = tf.range(tf.shape(inputs)[-1])
    embedded_words = self.word_embedding_layer(inputs)
    embedded_indices = self.position_embedding_layer(position_indices)
    return embedded_words + embedded_indices

def transformer_block(input_layer: layers.Layer, n: int):
  inp_norm = LayerNormalization(name=f"InitialNorm-{n}")(input_layer)

  # Multi-Headed Attention
  q = Dense(ATTENTION_KEY_SIZE, name=f"Q-{n}")(inp_norm)
  k = Dense(ATTENTION_KEY_SIZE, name=f"K-{n}")(inp_norm)
  v = Dense(ATTENTION_VALUE_SIZE, name=f"V-{n}")(inp_norm)
  
  attention = MultiHeadAttention(
      num_heads=ATTENTION_HEADS,
      key_dim=ATTENTION_KEY_SIZE,
      value_dim=ATTENTION_VALUE_SIZE,
      dropout=ATTENTION_DROPOUT,
      output_shape=(EMBEDDING_SIZE,),
      name=f"MHAttention-{n}"
  )(
      q, k, v,
      use_causal_mask=False
  )
  
  # First Residual Connection
  res_conn1 = Add(name=f"FirstAdd-{n}")([attention, input_layer])
  norm = LayerNormalization(name=f"FirstNorm-{n}")(res_conn1)
  
  # Linear Transformation
  lin1 = Dense(BASE_LINEAR_SIZE, activation="gelu", name=f"GeLU-{n}")(norm)
  lin2 = Dense(EMBEDDING_SIZE, name=f"OutLinear-{n}")(lin1)
  drop = Dropout(BASE_DROPOUT, name=f"BlockDropout-{n}")(lin2)
  
  # Second Residual Connection
  res_conn2 = Add(name=f"OutAdd-{n}")([drop, norm])
  return res_conn2

print("Building model")
# Build model
inp = Input(shape=(LOOKBACK,), name="Input")

embedding = PositionEmbeddingFixedWeights(LOOKBACK, VOCAB_SIZE, EMBEDDING_SIZE, name="Embeddings")(inp)
dropout = Dropout(BASE_DROPOUT, name="InitialDropout")(embedding)

block_input = transformer_block(dropout, 1)
for i in range(N_BLOCKS - 1):
    block_input = transformer_block(block_input, i + 2)

flat = Flatten(name="Flatten")(block_input)
output = Dense(VOCAB_SIZE, activation="softmax", name="Output")(flat)
model = models.Model(inp, output, name="MoGPT")

print("Compiling model")
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Training model. This will take a few hours, and may crash if you don't have enough GPU RAM. If you don't have NVIDIA CUDA installed and are training this on a CPU, this will take forever (i.e. days). Run on a Kaggle Cloud P100 notebook for better performance.")
model.fit(
    X, y,
    epochs=TRAIN_N_EPOCHS,
    batch_size=TRAIN_SIZE_BATCH,
    validation_split=0.1,
)

print("Saving model...")
model.save_weights("./gpt.h5")

print("Saving metadata...")
import json

def save_hyperparameters():
  with open("./meta.json", "w") as file:
    json.dump({
        "EMBEDDING_SIZE": EMBEDDING_SIZE,
        "N_BLOCKS": N_BLOCKS,
        "BASE_DROPOUT": BASE_DROPOUT,
        "BASE_LINEAR_SIZE": BASE_LINEAR_SIZE,
        "ATTENTION_HEADS": ATTENTION_HEADS,
        "ATTENTION_KEY_SIZE": ATTENTION_KEY_SIZE,
        "ATTENTION_VALUE_SIZE": ATTENTION_VALUE_SIZE,
        "ATTENTION_DROPOUT": ATTENTION_DROPOUT,
        "ATTENTION_MASK": ATTENTION_MASK,
        "LOOKBACK": LOOKBACK,
        "VOCAB_SIZE": VOCAB_SIZE
    }, file)
save_hyperparameters()
print("Saved")

def scale_array(array):
    """Scales an array linearly so that it sums to one.

    Args:
        array: The array to scale.

    Returns:
        A scaled version of the array.
    """

    sum_of_array = np.abs(np.sum(array))
    scaled_array = array / sum_of_array
    return scaled_array

def predict(string: str, n: int, greedy: bool=False) -> str:
    original = string

    if len(string) > LOOKBACK:
        string = string[-LOOKBACK:]
    elif len(string) < LOOKBACK:
        string = string.rjust(LOOKBACK)

    output = ""
    choices = list(range(len(tokens)))
    inp = encode(string)
    for i in tqdm.trange(n):
        logits = model.predict(inp.reshape(1, *inp.shape), verbose=0)
        logits = scale_array(np.log1p(logits))

        if greedy:
            out = np.argmax(choices)
        else:
            out = np.random.choice(choices, p=logits.reshape(-1))

        output += tokens[int(out)]

        inp = inp.tolist()
        inp.append(out)
        inp.pop(0)
        inp = np.array(inp)

    return original + output, output

PROMPT = "Hello, I am "
p, o = predict(PROMPT[-LOOKBACK:], 100)
print(p)
print(o)
