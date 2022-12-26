# Spotify Music Recommendation System
Many users find it very difficult to find good and new music recommendations that cater to their taste. I built a system to recommend music according to a user’s playlist. What makes this model unique is that it recommends music placing emphasis on the features of the songs rather than just recommending songs of artists that they already listen to. This enables users to find and explore new music and artists catered to their taste.

I used the spotify dataset and the artist dataset which I have attached in the data.zip file
This is a dataset which consists of songs in spotify from 1921 - 2020 and all features of the songs such as danceability, valence, acousticness, energy, popularity, etc.

## Correlation between various features of the dataset 

![output](https://user-images.githubusercontent.com/114499776/209531042-9f5a1c17-12ea-4417-8594-ed2f7c5557d2.png)

## Most Popular Songs each year (2000 - 2020)

![newplot](https://user-images.githubusercontent.com/114499776/209531558-4172c28b-e7a9-480d-a260-081907f9548f.png)

## KMeans Model

I found the optimal number of clusters for the given datset by the elbow method on plotting the within cluster sum of squares for k from 2 to 25. I found the ideal number of clusters k to be 6.

![output](https://user-images.githubusercontent.com/114499776/209531988-84ee8946-985c-4668-a02c-805c81d7d054.png)

### Song clustering

Clustering helps recognise patterns in the data. In order to do so, I decided to make use of a pipeline that consist of StandardScaler to normalise the data and K means to cluster the normalised data. Here is an example of how the code looks like.

```
song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=6, 
                                   verbose=False))
                                 ], verbose=False)
          
X = spotifydata.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
spotifydata['cluster_label'] = song_cluster_labels
```
I used PCA to visualise the large and complex spotifydata in much more simpler 2D format with different distinct colours representing clusters of data points.

![output](https://user-images.githubusercontent.com/114499776/209534567-a8c73725-27e2-46d7-b3fe-c1bb404c9284.png)

### Artist clustering

Similarly I clustered the artists together as shown and visualised it using plotly: 

![newplot](https://user-images.githubusercontent.com/114499776/209534814-a237e1de-5be4-46c9-b6a8-70482c646ebe.png)

## Spotify Recommendation System

We can now recommend music that caters to each individuals unique music taste by analysing the features of the songs/artists from the user's playlist.
The artist recommendation code is given below:

```
def recommend_artists(song_list, spotify_data, n_songs=10):
    artist_columns =  ['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 'liveness',
    'loudness', 'speechiness', 'tempo', 'valence', 'popularity', 'key']
    song_center = get_mean_vector(song_list, spotify_data, artist_columns)
    scaler = artist_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(artist_data[artist_columns])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])
    rec_artists = artist_data.iloc[index]
    return list(rec_artists['artists'])
```

We first determine the mean value of all the features of the songs in the user’s playlist and then normalize the data using Standard scaler in the scaler pipeline. Next, we pass all songs/artists of the dataset to the scaler pipeline where we normalize the data using Standardscaler() and cluster songs using Kmeans. Finally, we compare the clustered data and the mean value of songs/artists which was normalized and obtain 10 songs/artist closest to this mean value of our playlist using cdist.






