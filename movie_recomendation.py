# Importing the dataset
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

ratings.head()

# Poivoting the dataset
userRatings = ratings.pivot_table(index=['user_id'], columns=['title'], values='rating')
userRatings.head()


# Making the Correlation Matrix
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()


# Choosing a random user for recommendation
myRatings = userRatings.loc[0].dropna()
myRatings

# Making the recomendation using correlation matrix
simCandidates = pd.Series()
for i in range(0, len(myRatings.index)):
    print ("Adding sims for " + myRatings.index[i] + "...")
    # Retrieve similar movies to this one that I rated
    sims = corrMatrix[myRatings.index[i]].dropna()
    # Now scale its similarity by how well I rated this movie
    sims = sims.map(lambda x: x * myRatings[i])
    # Add the score to the list of similarity candidates
    simCandidates = simCandidates.append(sims)


# Grouping the same recomneded movie
simCandidates = simCandidates.groupby(simCandidates.index).sum()

#Sorting the values
simCandidates.sort_values(inplace=True, ascending=False)

# Removing the movies the user has already seen
filterSims = simCandidates.drop(myRatings.index)
filterSims.head(10)