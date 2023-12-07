#!/usr/bin/env python
# coding: utf-8

# # Generating Text with Neural Networks

# # Framing the Problem
# 
# ### What does the program do?
# 
# ### Where would this program be used?

# # Getting the Data

# ### Getting the Dataset Online and Reading the File
# The code below is first importing the tensorflow library and giving it the name of 'tf'. This is important so we do not have to keep typing out 'tensorflow' and can give it a short nickname.
# 
# Next, the code below shows retrieving the data from the homl.info/shakespeare website. The code is using keras from tensorflow.
# 
# The code is next reading the text from the file (given the nickname f) and putting the text from the file into the variable shakespear_text.

# In[3]:


import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()


# ### Printing Sections of the Text
# 
# The cell below just prints the first 80 characters of the text in the file. For example if you wanted to print the last characters of the text in the file you would code:
#                    "print(shakespeare_text[-80:])"

# In[4]:


print(shakespeare_text[:80]) # not relevant to machine learning but relevant to exploring the data


# # Preparing the Data

# ### Text Vectorisation
# (Source used: https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization)
# 
# The cell below will instantisate a TextVectorization in the variable text_vec_layer. This just means any text the text_vec_layer is used on will split it up by character (split="character") and will put all the characters into lowercase (standardize="lower").
# 
# text_vec_layer is then used to adapt our Shakespeare text and apply these rules to it.

# In[5]:


text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]


# This cell below will print out the text vectorisation. Each number represents a character in the text. The shape shows how many characters the are. And it shows that the data type of the vector is integers.

# In[6]:


print(text_vec_layer([shakespeare_text]))


# The cell below is just showing us how many tokens there are, which is the number of distinct characters, which in this case is 39.
# 
# It is also showing us the total number of characters.

# In[7]:


encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded)  # total number of chars = 1,115,394
print(encoded)


# In[8]:


print(n_tokens, dataset_size)


# ### Preparing Training and Test Data
# (Source used: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices, https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/op/data/WindowDataset
# https://stackoverflow.com/questions/71211053/what-tensorflows-flat-map-window-batch-does-to-a-dataset-array
# https://www.w3schools.com/python/ref_random_seed.asp#:~:text=Definition%20and%20Usage,uses%20the%20current%20system%20time.https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)
# 
# The cell below is creating a function that takes in a sequence, this will be the characters in the text. It also takes in the length.
# 
# The variable ds is created which is splitting up the sequence into slices which will be each of the characters.
# 
# Then .window is used which means each window is a dataset that contains a subset of elements from the sequence. It takes the arguments length+1, the shift=1 and drop_remainder is True. The length is how many elements from the dataset to put into a window. The shift means the input elements in the window shift by one. If the drop_remainder is True this means if the last window is smaller than the window size then it is dropped.
# 
# Flat_map makes sure the dataset is kept in the same order. The function is used to flatten the data from a dataset of datasets into a dataset of elements.
# 
# If shuffle = True then the dataset is shuffled. This works by having 100000 as the buffer size, which is just the amount of data used to shuffle, the seed is also set as equal to seed which is set to make sure the random numbers re the same if you rerun the program.
# 
# Dataset is then put into batches of size 32, which means amount of consecutive elements to combine into a single batch. A batch defines the number of samples before updating the model parameters.
# 
# Finally, the function returns the dataset.

# In[9]:


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(1000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# The cell below simply plugs in the data for the training set, the valid set, and the test set.

# In[10]:


length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:100_000], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[100_000:160_000], length=length)
test_set = to_dataset(encoded[160_000:], length=length)


# # Building and Training the Model

# In[11]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# In[12]:


shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# In[13]:


y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)  # choose the most probable character ID
text_vec_layer.get_vocabulary()[y_pred + 2]


# In[14]:


log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples


# In[15]:


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


# In[16]:


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


# In[17]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# In[18]:


print(extend_text("To be or not to be", temperature=0.01))


# In[19]:


print(extend_text("To be or not to be", temperature=1))


# In[20]:


print(extend_text("To be or not to be", temperature=100))


# In[ ]:




