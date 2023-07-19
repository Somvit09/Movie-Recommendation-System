# Movie-Recommendation-System
Movie Recommendation System using ML.

The objective of the code is to build a movie recommendation system using cosine similarity. 
It processes movie data from two CSV files (credits.csv and movies.csv) and creates a suitable 
dataset by merging the two dataframes. The text data is preprocessed to convert it into a compatible 
format for analysis.

Libraries used in the code:

    pandas: Used for data manipulation and DataFrame operations.

    numpy: Used for numerical operations and array processing.

    scikit-learn: Used for machine learning tasks and features CountVectorizer and cosine_similarity.

    ast: Used for safely evaluating literals, particularly to convert string representation of 
    lists/dictionaries to actual lists/dictionaries.

    nltk: Used for natural language processing tasks, particularly for stemming words using the 
    PorterStemmer.

The code uses CountVectorizer to convert the text data in the 'tags' column of the dataframe into 
numerical feature representations (vectors). The PorterStemmer is used to reduce words to their 
root form for better comparison. Cosine similarity is then computed between the vectors to find 
similar movies based on their 'tags'. The final result is a movie recommendation function, 
Recommend_Movies(movie), which takes a movie title as input and recommends similar movies using 
cosine similarity scores.

To recommend similar movies for a given movie title, the function finds the cosine similarity 
scores of the given movie with all other movies and sorts them in descending order. It then selects 
the top 10 similar movies (excluding the given movie itself) based on the highest cosine similarity 
scores. The function prints the titles of the recommended movies.

Overall, the code provides a basic movie recommendation system based on text data similarity, 
allowing users to input a movie title and get similar movie recommendations.


![image](https://github.com/Somvit09/Movie-Recommendation-System/assets/91347841/fc4a998f-aff6-4b57-8015-1448f2c13567)

![image](https://github.com/Somvit09/Movie-Recommendation-System/assets/91347841/b99feeb9-0bb5-4211-a3a5-248b97a1550c)


