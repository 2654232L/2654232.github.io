#!/usr/bin/env python
# coding: utf-8

# # Critically Engaging with AI Ethics
# 
# In this lab we will be critically engaging with existing datasets that have been used to address ethics in AI. In particular, we will explore the [**Jigsaw Toxic Comment Classification Challenge**](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge). This challenge brought to light bias in the data that sparked the [Jigsaw Unintended Bias in Toxicity Classification Challenge](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification). 
# 
# In this lab, we will dig into the dataset ourselves to explore the biases. We will further explore other datasets to expand our thinking about bias and fairness in AI in relation to aspects such as demography and equal opportunity as well as performance and group unawareness of the model. We will learn more about that in the tutorial below.
# 
# # Task 1: README!
# 
# This week, coding activity will be minimal, if any. However, as always, you will be expected to incorporate your analysis, thoughts and discussions into your notebooks as markdown cells, so I recommend you start up your Jupyter notebook in advance. As always, **remember**:
# 
# - To ensure you have all the necessary Python libraries/packages for running code you are recommended to use your environment set up on the **Glasgow Anywhere Student Desktop**.
# - Start anaconda, and launch Jupyter Notebook from within Anaconda**. If you run Jupyter Notebook without going through Anaconda, you might not have access to the packages installed on Anaconda.
# - If you run Anaconda or Jupyter Notebook on a local lab computer, there is no guarantee that these will work properly, that the packages will be available, or that you will have permission to install the extra packages yourself.
# - You can set up Anaconda on your own computer with the necessary libraries/packages. Please check how to set up a new environement in Anaconda and review the minimum list of Python libraries/packages, all discussed in Week 4 lab.
# - We strongly recommend that you save your notebooks in the folder you made in Week 1 exercise, which should have been created in the University of Glasgow One Drive - **do not confuse this with personal and other organisational One Drives**. Saving a copy of your notebooks on the University One Drive ensures that it is backed up (the first principles of digital preservation and information mnagement).
# - When you are on the Remote desktop, the `University of Glasgow One Drive` should be visible in the home directory of the Jupyter Notebook. Other machines may require additional set up and/or navigation for One Drive to be directly accessible from Jupyter Notebook.
# 

# # Task 2: Identifying Bias
# 
# This week we will make use of one of the [Kaggle](https://www.kaggle.com) tutorials and their associated notebooks to learn how to identify different types of bias. Biases can creep in at any stage of the AI task, from data collection methods, how we split/organise the test set, different algorithms, how the results are interpreted and deployed. Some of these topics have been extensively discussed and as a response, Kaggle has developed a course on AI ethics:
# 
# - Navigate to the [Kaggle tutorial on Identifying Bias in AI](https://www.kaggle.com/code/alexisbcook/identifying-bias-in-ai/tutorial). 
# - In this section we will explore the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) to discover different types of biases that might emerge in the dataset. 
# 
# #### Task 2-a: Understanding the Scope of Bias
# 
# Read through the first page of the [Kaggle tutorial on Identifying Bias in AI] to understand the scope of biases discussed at Kaggle.
# - How many types of biases are described on the page? 
# - Which type of bias did you know about already before this course and which type was new to you? 
# - Can you think of any others? Create a markdown cell below to discuss your thoughts on these questions.
# 
# Note that the biases discussed in the tutorial are not an exhaustive list. Recall that biases can exist across the entire machine learning pipeline. 
# 
# - Scroll down to the end of the Kaggle tutorial page and click on the link to the exercise to work directly with a model and explore the data.** 
# 
# #### Task 2-b: Run through the tutorial. Take selected screenshorts of your activity while doing the tutorial.
# 
# - Discuss with your peer group, your findings about the biases in the data, including types of biases. 
# - Demonstrate your discussion with examples and screenshots of your activity on the tutorial. Present these in your own notebook.
# 
# Modify the markdown cell below to address the Tasks 2-a and 2-b.

# ## <span style="color: red;">Task 2a</span>
# 
# 1. How many types of bias are described on the page? There are six types of bias.
# 2. Which type of bias did you know about already before this course and which type was new to you? I already knew about representation bias and historial bias, the other 4 were new to me.
# 3. Can you think of any others? I know of sampling bias, where a sample is not truly representative.
# 
# ## <span style="color: red;">Task 2b</span>
# 
# ![Screenshot of Sample of Model](AI_Ethics\images\kaggle1.png)
# This screenshot shows the sample of the model, and how it works. It classifies comments into 'Toxic' or 'Non Toxic'
# 
# ![Screenshot of Accuracy of Model](AI_Ethics\images\kaggle2.png)
# This screenshot shows the calculated accuracy of the model
# 
# ![Screenshot of Toxic Comment](AI_Ethics\images\kaggle3.png)
# This screenshot shows the calculated accuracy of the model
# 
# ![Screenshot of Non Toxic Comment](AI_Ethics\images\kaggle4.png)
# This screenshot shows the calculated accuracy of the model
# 
# ![Screenshot of Toxic Words](AI_Ethics\images\kaggle5.png)
# This screenshot shows the list of toxic words in the model.
# 
# ![Screenshot of Accuracy of Model](AI_Ethics\images\kaggle6.png)
# ![Screenshot of Accuracy of Model](AI_Ethics\images\kaggle7.png)
# Shows the bias in the model. This is biased because it says that some identities are toxic whilst others are not.
# 
# ### <span style="color: red;">From Exericse 5:</span>
# "You notice that comments that refer to Islam are more likely to be toxic than comments that refer to other religions, because the online community is islamophobic. What type of bias can this introduce to your model?"
# 
# This shows historical bias because of the 'flawed state' of the online world where the data was collected.
# 
# ### <span style="color: red;">From Exercise 6:</span>
# "You take any comments that are not already in English and translate them to English with a separate tool. Then, you treat all posts as if they were originally expressed in English. What type of bias will your model suffer from?"
# 
# This shows measurement bias, and could show aggregation bias. Measurement bias because the comments can get lost in translation.
# 
# ### <span style="color: red;">From Exercise 7:</span>
# "The dataset you're using to train the model contains comments primarily from users based in the United Kingdom.
# After training a model, you evaluate its performance with another dataset of comments, also primarily from users based in the United Kingdom -- and it gets great performance! You deploy it for a company based in Australia, and it does not perform well, because of differences between British and Australian English. What types of bias does the model suffer from?"
# 
# This shows deployment and evaluation bias. This is because the information is from the UK but is being shown to people in Australia. There is also representation bias because of this.
# 
# ### <span style="color: red;">My Findings</span>
# From this exercise I have found out more about biases in models and where they can come from. I have also learnt more about different types of bias in more depth and have been shown examples of them.

# # Task 3: Large Language Models and Bias: Word Embedding Demo
# 
# Go to the [embedding projector at tensorflow.org](http://projector.tensorflow.org/). This may take some time to load so be patient! There is a lot of information being visualised. This will take especially long if you select "Word2Vec All" as your dataset. The projector provides a visualisation of the langauge language model called **Word2Vec**.
# 
# This tool also provides the option of visualising the organisation of hand written digits from the MNIST dataset to see how data representations of the digits are clustered together or not. There is also the option of visualising the `iris` dataset from `scikit-learn` with respect to their categories. Feel free to explore these as well if you like.
# 
# For the current exercise, we will concentrate on exploring the relationships between the words in the **Word2Vec** model. First, select **Word2Vec 10K** from the drop down menu (top lefthand side). This is a reduced version of **Word2Vec All**. You can search for words by submitting them in the search box on the right hand side. 
# 
# #### Task 3.1: Initial exploration of words and relationships
# 
# - Type `apple` and click on `Isolate 101 ppints`. This reduces the noise. Note how juice, fruit, wine are closer together than macintosh, computers and atari. 
# - Try also words like `silver` and `sound`. What are your observations. Does it seem like words related to each other are sitting closer to each other?
# 
# #### Task 3.2: Exploring "Word2Vec All" for patterns
# 
# - Try to load "Word2Vec All" dataset if you can (this may take a while so be patient!) and explore the word `engineer`, `drummer`or any other occupation - what do you find? 
# - Do you think perhaps there are concerns of gender bias? If so, how? If not, why not? Discuss it with our peer group and present the results in a your notebook.
# - Why not make some screenshots to embed into your notebook along with your comment? This could make it more understandable to a broader audience. 
# - Do not forget to include attribution to the authors of the Projector demo.
# 
# Modify the markdown cell below to present your thoughts.

# ## <span style="color: red;">Task 3.1</span>
# 
# Juice and fruit and wine are close together. But macintosh and computers are far away from each other.
# With silver when you isolate 101 points, they all apear to be different elements and elements like gold and copper are closest, which is to be expected because they are close to each other on the periodic table.
# With sound the closest points are also related to sound, examples include: noise, sounds, heard, and listening.
# 
# ## <span style="color: red;">Task 3.2</span>
# 
# When using the word engineer, the closest words are engineers, engineering, scientist, and designer. These are all relevant words. I have not seen any gender bias, this is probably because engineer is a gender neutral name for an occupation. Speaking with peers they also think the same thing.
# 
# ![Screenshot of Embedding projector on the word engineer](AI_Ethics\images\embeddingprojector_engineer.png)
# #### Screenshot from https://projector.tensorflow.org/

# # Task 4: Thinking about AI Fairness 
# 
# So we now know that AI models (e.g. large language models) can be biased. We saw that with the embedding projector already. We discussed in the previous exercise about the machine learning pipeline, how the assessment of datasets can be crucicial to deciding the suitability of deploying AI in the real world. This is where data connects to questions of fairness.
# 
# - Navigate to the [Kaggle Tutorial on AI Fairness](https://www.kaggle.com/code/alexisbcook/ai-fairness). 
# 
# #### Task 4-a: Topics in AI Fairness
# Read through the page to understand the scope of the fairness criteria discussed at Kaggle. Just as we dicussed with bias, the fairness criteria discussed at Kaggle is not exhaustive. 
# - How many criteria are described on the page? 
# - Which criteria did you know about already before this course and which, if any, was new to you? 
# - Can you think of any other criteria? Create a markdown cell and note down your discussion with your peer group on these questions.
# 
# #### Task 4-b: AI fairness in the context of the credit card dataset. 
# Scroll down to the end of [the page on AI fairness](https://www.kaggle.com/code/alexisbcook/ai-fairness) to find a link to another interactive exercise to run code in a notebook using credit card application data.
# - Run the tutorial, while taking selected screenshots.
# - Discuss your findings with your peer group.
# - Note down the key points of your activity and discussion in your notebook using the example and screenshots of your activity on the tutorial.
# 
# 
# Report the results of the activity and discussion by modifying the markdown cell below.

# ## <span style="color: red;">Task 4a</span>
# 
# Four fairness criteria are described on the page.
# I didn't know the names of any of the fairness criteria, I recognised the method of equal oppurtunity and demographic parity but was unaware of the names.
# I did not know any other types of fairness criteria.
# 
# ## <span style="color: red;">Task 4b</span>
# 
# ![Screenshot of first 5 lines of dataset](AI_Ethics\images\kagglefairness1.png)
# This screenshot shows the first five lines of the dataset for the credit card approval model
# 
# ![Screenshot of descriptions of dataset](AI_Ethics\images\kagglefairness2.png)
# This photo describes what each of the headings mean on the dataset.
# 
# ![Screenshot of results of the model](AI_Ethics\images\kagglefairness3.png)
# ![Screenshot of group a confusion matrix](AI_Ethics\images\kagglefairness4.png)
# ![Screenshot of group b confusion matrix](AI_Ethics\images\kagglefairness5.png)
# These screenshots show the results of the model and how many approvals were given. The photos show the confusions matrices for Group A and for Group B.
# 
# ![Screenshot of fairness criteria in the model](AI_Ethics\images\kagglefairness6.png)
# This photo shows how all the fairness criteria is in favor of Group B. And shows that if you are in group A and should be approved that your chances of actually being approved are very low.
# 
# ![Screenshot of Flowchart of how model works](AI_Ethics\images\kagglefairness7.png)
# This photo shows the flowchart which depicts how the model for credit card approvals works.
# 
# ![Screenshot describing how the flowchart works](AI_Ethics\images\kagglefairness8.png)
# This photo describes how the flowchart work and how the model decides whether to approve or deny a credit card.
# 
# ![Screenshot of description of unfairness in the model](AI_Ethics\images\kagglefairness9.png)
# This photo describes the source of unfairness in the model, which is that group a and group b are not treated fairly and are not treated equally.
# 
# ![Screenshot of new model](AI_Ethics\images\kagglefairness10.png)
# Shows a new version of the model in hopes of making it more fair.
# 
# ![Screenshot of descriptions of new results of model](AI_Ethics\images\kagglefairness11.png)
# Describes how the new model has dropped in accuracy. The new model is still biased towards group B, but in terms of equal oppurtunity and accuracy, it is now in favor of group A.
# 
# ![Screenshot of third model and its results](AI_Ethics\images\kagglefairness12.png)
# This photo shows a third model and its results. 
# 
# ![Screenshot of evaluation of results of third model](AI_Ethics\images\kagglefairness13.png)
# This photo evaluates the results of the third model. There is an overall drop in accuracy but the two groups are more close. This model is overall more fair than the others for demographic parity, but is more biased in favor of group A for equal oppurtunity and accuracy.

# # Task 5: AI and Explainability
# 
# In this section we will explore the reasons behind decisions that AI makes. While this is really hard to know, there are some approaches developed to know which features in your data (e.g. median_income in the housing dataset we used before) played a more important role than others in determining how your machine learning model performs. One of the many approaches for assessing feature importance is **permutation importance**.
# 
# The idea behind permutation importance is simple. Features are what you might consider the columns in a tabulated dataset, such as that might be found in a spreadsheet. 
# - The idea of permutation importance is that a feature is important if the performance of your AI program gets messed up by **shuffling** or **permuting** the order of values in that feature column for the entries in your test data. 
# - The more your AI performance gets messed up in response to the shuffling, the more likely the feature was important for the AI model.
#  
# To make this idea more concrete, read through the page at the [Tutorial on Permutation Importance](https://www.kaggle.com/code/dansbecker/permutation-importance) at Kaggle. The page describes an example to "predict a person's height when they become 20 years old, using data that is available at age 10". 
# 
# The page invites you to work with code to calculate the permutation importance of features for an example in football to predict "whether a soccer/football team will have the "Man of the Game" winner based on the team's statistics". Scroll down to the end of the page to the section "Your Turn" where you will find a link to an exercise to try it yourself to calculate the importance of features in a Taxi Fare Prediction dataset.
# 
# #### Task 1-a: Carry out the exercise, taking screenshots of the exercise as you make progress. Using screen shots and text in your notebook, answer the following question: 
# 1. How many features are in this dataset? 
# 2. Were the results of doing the exercise contrary to intuition? If yes, why? If no, why not? 
# 3. Discuss your results with your peer group.
# 4. Include your screenshots, text, and discyssions in a markdown cell.
# 
# #### Task 1-b: Reflecting on Permutation Importance.
# 
# - Do you think the permutation importance is a reasonable measure of feature importance? 
# - Can you think of any examples where this would have issues? 
# - Discuss these questions in your notebook - describe your example, if you have any, and discuss the issues. 

# ## <span style="color: red;">Task 5a</span>
# 1. There are seven features of this dataset. Only five features are used in the first dataset.
# 2. Results were not different to what I expected. Distance was the most important factor I thought of to begin with and this was the most important feature in the end for calculating taxi fares.
# 
# #### <span style="color: red;">Question 1</span>
# Which features are useful for predicting taxi fares? I think all of the latitude and longitude data used in this model are useful. It needs to be known where the taxi is being picked up and dropped off to work out the distance and therefore the taxi fare. I do not think the passenger number is useful because that is not normally used in working out taxi fare.
# 
# #### <span style="color: red;">Question 2</span>
# ![picture of my answer for question 2](AI_Ethics\images\permutationimportance_question2.png)
# 
# #### <span style="color: red;">Question 3</span>
# Latitude may matter more because the latitude is more likely to change at a greater value than the longitude values. Latitude may also be worth more in taxi fares in the area.
# 
# #### <span style="color: red;">Question 4</span>
# ![picture of my code for question 4](AI_Ethics\images\permutationimportance_question4.png)
# 
# #### <span style="color: red;">Question 5</span>
# The scale of the features does not affect permuation importance in this model. The absolute change features here have high importance because they are the total distance travelled which is the most important factor in determining taxi fares.
# 
# #### <span style="color: red;">Question 6</span>
# We cannot tell from permuation importance results whether lat or longitude distance is more expensive. latitude distancs tend to be larger so that is why it could be more expensive.
# 
# ## <span style="color: red;">Task 5b</span>
# I think permuation importance is a good indicator of feature importance.
# There could be a problem if features have a strong correlation. This can come up with unlikely data instances. (Source: https://christophm.github.io/interpretable-ml-book/feature-importance.html).

# # Task 6: Further Activities for Broader Discussion
# 
# Apart from the [**Jigsaw Toxic Comment Classification Challenge**](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge) another challenge you might explore is the [**Inclusive Images Challenge**](https://www.kaggle.com/c/inclusive-images-challenge). Read at least one of the following.
# 
# - The [announcement of the Inclusive Images Challenge made by Google AI](https://ai.googleblog.com/2018/09/introducing-inclusive-images-competition.html). Explore the [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html) - this is where the Inclusive Images Challenge dataset comes from.
# - Article summarising [the Inclusive Image Challenge at NeurIPS 2018 conference](https://link.springer.com/chapter/10.1007/978-3-030-29135-8_6)
# - Explore the [recent controversy](https://www.theverge.com/21298762/face-depixelizer-ai-machine-learning-tool-pulse-stylegan-obama-bias) about bias in relation to [PULSE](https://paperswithcode.com/method/pulse) which, among other things, sharpens blurry images.
# - Given your exploration in the sections above, what problems might you foresee with [these tasks attempted with the Jigsaw dataset on toxicity](https://link.springer.com/chapter/10.1007/978-981-33-4367-2_81)?
# 
# There are many concepts (e.g. model cards and datasheets) omitted in discussion above about AI and Ethics. To acquire a foundational knowledge of transparency, accessibility and fairness:
# 
# - You are welcome to carry out the rest of the [Kaggle course on Intro to AI Ethics](https://www.kaggle.com/learn/intro-to-ai-ethics) to see some ideas from the Kaggle community. 
# - You are welcome to carry out the rest of the [Kaggle tutorial on explainability]( https://www.kaggle.com/learn/machine-learning-explainability) but these are a bit more technical in nature.

# ## <span style="color: red;">Task 6</span>
# What problems could arise from these tasks attempted with the Jigsaw dataset on toxicity?
# In the PULSE controversy there was a picture that turned Obama white. This shows the bias AI has that has been ingrained since the beginning. This controversy has shown the racial bias in AI. This is similar to the Jigsaw comment toxicity where there was racial bias in the comments, where it said 'I have a white friend' was labelled non-toxic but 'I have a black friend' was labelled as toxic. It appeared that PULSE was generating more pictures of white people than faces of people of colour.

# # Summary
# 
# In this lab, you explored a number of areas that pose challenges with regard to AI and ethics: bias, fairness and explainability. This, and other topics in reposible AI development, is currently at the forefront of the AI landscape. 
# 
# The discussions coming up in the lectures on applications of AI (to be presented by guest lecturers in the weeks to come) will undoubtedly intersect with these concerns. In preparation, you might think, in advance, about **what distinctive questions about ethics might arise in AI applications in law, language, finance, archives, generative AI and beyond**.   

# In[ ]:




