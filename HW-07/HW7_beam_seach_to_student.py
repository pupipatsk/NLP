# %% [markdown]
# # HW7: Beam Search Decoding - News Headline Generation

# %% [markdown]
# In this exercise, you are going to learn and implement decoding techniques for sequence generation. Usually, the sequence is generated word-by-word from a model. In each step, the model predicted the most likely word based on the predicted words in previous steps (this is called auto-regressive decoding).
#
# As such, it is very important how you decide on what to predicted at each step, as it will be conditioned on to predicted all of the following steps. We will implement two of main decoding techniques introduced in the lecture: **Greedy Decoding** and **Beam Search Decoding**. Greedy Decoding immediately chooses the word with best score at each step, while Beam Search Decoding focuses on the sequence that give the best score overall.
#
# To complete this exercise, you will need to complete the methods for decoding for a text generation model trained on [New York Times Comments and Headlines dataset](https://www.kaggle.com/aashita/nyt-comments). The model is trained to predict a headline for the news given seed text. You do not need to train any model model in this exercise as we provide both the pretrained model and dictionary.
#

# %% [markdown]
# ## Download model and vocab and setup

# %%
# !wget -O vocab.txt "https://www.dropbox.com/s/ht12ua9vpkep6l8/hw9_vocab.txt?dl=0"
# !wget -O model.bin "https://www.dropbox.com/s/okmri7cnd729rr5/hw9_model.bin?dl=0"

# %%
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

# %%
class RNNmodel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dropout_rate):

        super().__init__()
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, 128, num_layers=2,
                     batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, src):
        embedding = self.embedding(src)
        output,_ = self.rnn(embedding)
        output = self.dropout(output)
        prediction = self.fc2(output)
        return prediction

# %%
with open("vocab.txt") as f:
  vocab_file = f.readlines()
embedding_dim = 64
dropout_rate = 0.2

model = RNNmodel(len(vocab_file), embedding_dim, dropout_rate)
model.load_state_dict(torch.load("model.bin",map_location='cpu'))
model.eval()

# %%
vocab = [v.strip() for v in vocab_file]
vocab_size = len(vocab)
print(f"Vocab Size: {vocab_size}")
vocab[:10]

# %%
stoi = { ch:i for i,ch in enumerate(vocab) }
tokenizer = Tokenizer(WordLevel(stoi, unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()
tokenized_text = tokenizer.encode("the a of to unknowns")
print(tokenized_text)
print(tokenized_text.ids)
print(tokenized_text.tokens)
print(tokenizer.decode(tokenized_text.ids))

# %% [markdown]
# ## 1. TODO: Greedy decode
# Normally, in sequence generation task, the model will continue generating tokens until an end-of-sequence symbol appear or the maximum length is reached. For this task:
# - The end-of-sequence symbol is "< eos >" and its index is 2
# - Use the maximum generation length of 15

# %%
eos_token = '<eos>'
max_gen_length = 15

# %%
def greedy_decode(seed_text, tokenizer):
    """Greedy decodes with seed text.

        Args:
        seed_text: The seed string to be used as initial input to the model.
        tokenizer: The tokenizer for converting word to index and back.

        Your code should do the followings:
          1. Convert current_text to sequences of indices
          2. Predict the next token using the model and choose the token with the highest score as output
          3. Append the predicted index to current_text
          4. Loop until completion
          5. Return text prediction and a list of probabilities of each step

        You do not need to stop early when end-of-sequence token is generated and can continue decoding
        until max_gen_length is reached. We can filter the eos token out later.
    """

    return output,probs

# %%
def clean_output(text, eos_token):
    """Drop eos_token and every words that follow"""
    pass

# %%
sample_seeds = ["to", "america", "people", "next", "picture", "on"]
for seed in sample_seeds:
    pass

# %% [markdown]
# Your output should be:
#
# *   to encourage creativity in the new york bill
# *   america s lethal export
# *   people to balloon to make a criminal with a dog with a callous rival
# *   next phenom english clubs 2 call another deal in the same arrivals
# *   picture perfect chapter a spot of view of banning care
# *   on the catwalk in saudi arabia
#
#
#
#
#
#

# %% [markdown]
# ## 2. TODO: Beam search decode
#
# Another well-known decoding method is beam search decoding that focuses more on the overall sequence score.
#
# Instead of greedily choosing the token with the highest score for each step, beam search decoding expands all possible next tokens and keeps the __k__ most likely sequence at each step, where __k__ is a user-specified beam size. A sequence score is also calculated according user-specified cal_score() function.
# The beam with the highest score after the decoding process is done will be the output.

# %% [markdown]
# There are a few things that you need to know before implementing a beam search decoder:
# - When the eos token is produced, you can stop expanding that beam
# - However, the ended beams must be sorted together with active beams
# - The decoding ends when every beams are either ended or reached the maximum length, but for this task, you can continue decoding until the max_gen_len is reached
# - We usually work with probability in log scale to avoid numerical underflow. You should use np.log(score) before any calculation
# - **As probabilities for some classes will be very small, you must add a very small value to the score before taking log e.g np.log(prob + 0.00000001)**

# %% [markdown]
# #### Sequence Score
# The naive way to calculate the sequence score is to __multiply every token scores__ together. However, doing so will make the decoder prefer shorter sequence as you multiply the sequence score with a value between \[0,1\] for every tokens in the sequence. Thus, we usually normalize the sequence score with its length by calculating its __geometric mean__ instead.
#
# **You should do this in log scale**

# %%
def cal_score(score_list, length, normalized=False): #cal score for each beam from a list of probs

    if normalized:
        pass
    else:
        pass
    return

# %%
def beam_search_decode(seed_text, max_gen_len, tokenizer, beam_size=5, normalized=False):
    """We will do beam search decoing using seed text in this function.

    Output:
    beams: A list of top k beams after the decoding ended, each beam is a list of
      [seed_text, list of scores, length]

    Your code should do the followings:
    1.Loop until max_gen_len is reached.
    2.During each step, loop thorugh each beam and use it to predict the next word.
      If a beam is already ended, continues without expanding.
    3.Sort all hypotheses according to cal_score().
    4.Keep top k hypotheses to be used at the next step.
    """
    # For each beam we will store (generated text, list of scores, and current length, is_finished)
    # Add initial beam
    beams = [[[seed_text], [], 0, False]]
    for _ in range(max_gen_len):
      pass

    return beams

# %% [markdown]
# ## 3. Generate!
# Generate 6 sentences based on the given seed texts.
#
# Decode with the provided seed texts with beam_size 5, max_gen_len 10. Compare the results between greedy, normalized, and unnormalized decoding.
#
# Print the result using greedy decoding and top 2 results each using unnormalized and normalized decoing for each seed text.
#
# Also, print scores of each candidate according to cal_score(). Use normalization for greedy decoding.

# %%
sample_seeds = ["to", "america", "people", "next", "picture", "on"]
max_gen_len=10
for seed in sample_seeds:
    pass

# %% [markdown]
# Your output should be:
#
#
# ```
# -Greedy-
# to encourage creativity in the new york bill  0.13
# -Unnormalized-
# To Consult Exploring Recipes For New Jersey 0.00
# To Consult Exploring Recipes Up The Pacific Northwest 0.00
# -Normalized-
# To Consult Exploring Recipes Up The Pacific Northwest 0.17
# To Consult Exploring Recipes Up The Least Of The Week 0.16
#
# -Greedy-
# america s lethal export  0.25
# -Unnormalized-
# America S Lethal Export 0.02
# America S Desert Aisles 0.01
# -Normalized-
# America S Lethal Export 0.25
# America S Desert Aisles 0.20
#
# -Greedy-
# people to balloon to make a criminal with a dog with a callous rival  0.14
# -Unnormalized-
# People To Balloon For A Criminal 0.00
# People To Balloon For A Criminal With Trump 0.00
# -Normalized-
# People To Balloon For A Criminal With A Second Fiddle 0.13
# People To Balloon For A Criminal With Trump 0.13
#
# -Greedy-
# next phenom english clubs 2 call another deal in the same arrivals  0.12
# -Unnormalized-
# Next S Blist Revue 0.00
# Next Phenom English Clubs 1 A Chance To Be Back 0.00
# -Normalized-
# Next S Blist Revue 0.14
# Next Phenom English Clubs 1 A Chance To Be Back 0.14
#
# -Greedy-
# picture perfect chapter a spot of view of banning care  0.08
# -Unnormalized-
# Picture Perfect Use Coffee 0.00
# Picture Korean A Bonanza Of Pancakes 0.00
# -Normalized-
# Picture Korean A Bonanza Of Contemplation Times Of Trump S Son 0.12
# Picture Korean A Bonanza Of Pancakes 0.07
#
# -Greedy-
# on the catwalk in saudi arabia  0.19
# -Unnormalized-
# On The Billboard Chart 0.00
# On The Catwalk In Saudi Arabia 0.00
# -Normalized-
# On The Whole30 Diet Vowing To Eat Smarter Carbs To Be 0.27
# On The Whole30 Diet Vowing To Eat Smarter Carbs For Because 0.26
#
# ```
#
#

# %% [markdown]
# # Answer Questions in MyCourseVille!
#
# Use the seed word "usa" to answer questions in MCV.


