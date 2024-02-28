# Genre Categorizer (Midterm Report)
Project for "CS4641 - Machine learning" by Taylor Smith, Sichong Hua, Honghao Zhu, Ryan Choi, and Joshua Atler.

## Introduction
If you go on Spotify and listen to any number of songs, the Spotify algorithm will begin to recommend new music based on what you have previously listened to. This in itself is machine learning: where, based on existing data, you are able to make predictions as to what a user may want to listen to. Following this theme of music, we want to bridge the gap between lyrics and genres, in which given some lyrics, we can identify the genre of the song.
<br/>
<br/>
Since the initial proposal that we made, we have been attempting to create the training dataset and scrape data from the Genius API. However, since the Genius API does not provide genre data, we have been working to resolve this issue and figure out a way to associate genre with the song/song lyric data that we end up with. One solution to this problem is to assign the genre manually based on artists, which is one of our future goals, but for right now, it seems as though utilizing a new dataset that does provide genre data will be the best solution in terms of allowing us to make continual progress in the project.

## Problem Statement
With the large number of genres that currently exists, it can sometimes be challenging to categorize a certain song. As a team, we want to create an application that can understand the relationship between the lyrics and genres, with the lyrics being any meaningful text one can input. More specifically, we are going to first utilize an NLP technique2 to analyze components of lyrics in respect to its genre and train a classifier model to predict genre of the song. Combining the identified genre and component of lyrics, we then identify the sentiment level of the song.

## Method
The primary intended dataset for our project has changed from The Million Song Dataset1 (Bertin-Mahieux, 2011) to Sam Ho's universal song encoder lyric dataset. From this dataset, we were able to get the artist, song title, lyrics, song genre, lyric count, and lyric count norm. The first stage of the project is data parsing and cleaning by removing punctuations and relevant stop words using Python regular expression and NLTK package. Data will be collected and stored in a customized database(panda dataframe). Although lyrics count are not necessary for training the data, those datas will be used for our pretraining visualization. Later, we will mostly focus on the lyrics(input) and the genre(output).
<br/>
<br/>
Analyzing the various features we collected, we hope to be able to visualize the outputs that are relevant to our study. For the training stage, we plan to use regression methods and Gaussian Naive Bayes to calculate the probability of a lyric belonging to a genre given its features. We then make the prediction using MLE in order to determine which genre the song would most likely belong to. We will cross compare the result utilizing different methods in order to check for accuracy. Key classification metrics such as precision, recall, and F1-score will be used to run the accuracy test.  

## Data Collection

Instead of getting lyric data from the Genius API, we decided that it would be both easier and more efficient to use a pre-compiled lyric dataset from Sam Ho's song lyric universal song encoder. The training set includes data points representing the "Artist Name, Song Name, Lyrics, Genre, Lyrics count, Lyrics count norm" attributes. From this, we applied some data cleaning and pre-processing techniques, involving removing the punctuation from the lyrics, including commas and periods, and removing all stop words from the lyrics as well. The stop words that we removed were defined by the NLTK.corpus package in Python.
<br/>
<br/>
Our main goal is to train a model that can use text processing technique to analyze the correlation parameter between genre and lyrics. Since we have the artist name and song name, we will use these two attributes as a simple logic guide to see if our training set yields the right genre type(eg.Katy Perry will have the majority of her song under "pop" genre). 3600 songs are included in the training set. We understand that some of these will associations between artist and genre will have to be done manually.

## Results
The genre types are be represented as clusters. We have produced the histograms of the word clusters for each genre. The histograms plot Genre vs. Genre Counts, defining the number of songs identified to be a part of a certain genre. In addition, histograms like "Average Words Per Song by Genre" and "Top 10 Most Frequent Words for All Genres" are also being generated for pre-training visualization. We have also formed a word cloud for each genre, based on the most common words associated with a certain genre. The raw results would be the assignment and correlation between a song and a chosen genre set. <b>Dataset can be visualized in LyricsScrape Test Ground.ipynb as lyricsData, accompany with our data cleaning process, histograms and word clouds. </b>



## References
#### 1.Thierry Bertin-Mahieux, Daniel P.W. Ellis, Brian Whitman, and Paul Lamere. The Million Song Dataset. In Proceedings of the 12th International Society for Music Information Retrieval Conference (ISMIR 2011), 2011.
#### 2. Chu Claudia, Kumala Enoch, 2020, Mapping song lyrics to musical feature space, https://suraj-masand.github.io/cs4641-project/index.html
#### 3. Tsaptsinos, Alexandros, 2017, Lyrics-Based Music Genre Classification Using a Hierarchical Attention Network
#### 4. Songlyrics_univeral_sentence_encoder by kitsamho, https://github.com/kitsamho/songlyrics_univeral_sentence_encoder
