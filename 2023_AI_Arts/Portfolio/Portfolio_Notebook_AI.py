#!/usr/bin/env python
# coding: utf-8

# # Week One: Getting Started with Anaconda, Jupyter Notebook and Python
# 
# Exercises to familiarise myself with Jupyter Notebook and its relationship to Python

# ### Why I chose to join this course
# 
# I chose to join this course because I have an interest in AI and its development.

# ### Prior Experience
# 
# I have used Python since I was in high school and during my first two years at university doing computer science courses. I do not have any experience with AI.

# ### What I expect to learn from this course
# 
# - About recent AI developments
# - How AI is expected to develop in the future
# - How to use some different python libraries
# - How to write a literature review

# ### Task 1.4

# In[1]:


print("Hello World!")


# In[2]:


message = "Hello World!"

print(message)


# ### Task 1.5

# In[5]:


message = "Hello my name is Hannah"

print(message)
print(message + message)
print(message*3)
print(message[0])


# message + message outputs the message twice one after the other
# 
# message * 3 outputs the message three times
# 
# message[0] output the first character of the string
# 
# I think message is an adequate variable name for the content it has. It could be made better by being called something more specific, for example, greeting.

# ### Task 1.6

# In[6]:


from IPython.display import *


# In[7]:


YouTubeVideo("8MgjyEAl2OA&ab_c")


# ### Task 1.8

# In[8]:


import webbrowser
import requests

print("Shall we hunt down an old website?")

site = input("Type a website URL: ")

era = input("Type year, month, and date, e.g. 20150613: ")

url = "http://archive.org/wayback/available?url=%s&timestamp=%s" % (site, era)

response = requests.get(url)

data = response.json()

try:
    old_site = data["archived_snapshots"]["closest"]["url"]
    print("Found this copy: ", old_site)
    print("It should appear in your browser.")
    webbrowser.open(old_site)
except:
    print("Sorry, could not find the site.")


# # Week Two: Exploring Data in Multiple Ways

# In[1]:


from IPython.display import Image


# In[2]:


Image ("picture1.jpg")


# In[3]:


from IPython.display import Audio


# In[10]:


Audio ("audio1.mid")


# In[11]:


Audio ("audio2.ogg")
#This file is licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license.You are free: 
#•	to share – to copy, distribute and transmit the work
#•	to remix – to adapt the work
#Under the following conditions: 
#•	attribution – You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
#•	share alike – If you remix, transform, or build upon the material, you must distribute your contributions under the same or compatible license as the original.
#The original ogg file was found at the url: 
#https://en.wikipedia.org/wiki/File:GoldbergVariations_MehmetOkonsar-1of3_Var1to10.ogg


# Audio 2 does not play because the file is not called audio. Once the name has been changed to match it works.

# ## Task 3.2 matplotlib Library to Look at the Numerical Data of a Picture

# In[13]:


from matplotlib import pyplot
test_picture = pyplot.imread("picture1.jpg")
print("Numpy array of the image is: ", test_picture)
pyplot.imshow(test_picture)


# In[14]:


test_picture_filtered = 2*test_picture/3
pyplot.imshow(test_picture_filtered)


# The picture has been filtered. I think the colours of the original array have been changed by multiplying the colour by 2 and dividing by 3 to create this edited photo.

# ## Task 3.2 Exploring scikit-learn

# In[15]:


from sklearn import datasets
dir(datasets)


# I chose load_iris, load_boston, load_digits. These sounded most interesting to me and I want to find out what is in these datasets.

# In[19]:


iris_data = datasets.load_iris()
digits_data = datasets.load_digits()
boston_data = datasets.load_boston()

print(iris_data.DESCR)
print(boston_data.DESCR)
print(digits_data.DESCR)


# In[20]:


boston_data.feature_names


# The Boston data has 13 features.

# In[22]:


iris_data.target_names


# The iris data has 3 target names. I think these are each different types of Iris plant the dataset is about.

# In[24]:


digits_data.target_names


# In[25]:


digits_data.feature_names


# boston_data.target_names

# The boston data does not have any target names.

# In[27]:


boston_data.keys()


# ## Task 3.4 Basic Data Exploration with Pandas library

# In[35]:


from sklearn import datasets
import pandas

wine_data = datasets.load_wine()

wine_dataframe = pandas.DataFrame(data=wine_data['data'], columns = wine_data['feature_names'])

wine_dataframe.head()
wine_dataframe.describe()


# The head() command gets the first 5 rows of the data frame. describe() adds the row labels e.g. count, mean, etc.

# ## Task 3.5 Thinking about Data Bias
# 
# To assess bias I would look for number of features explore, the amount of references used, and the number of sets used - the bigger the number the better.
