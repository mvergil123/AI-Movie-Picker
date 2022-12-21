import tensorflow as tf
import pandas as pd

# Load the movie data into a Pandas DataFrame
movies = pd.read_csv("movies.csv", names=["movieId", "title", "genres"])

# Preprocess the data by one-hot encoding the genres
movies = pd.get_dummies(movies, columns=["genres"])

# Define the input and output data
x = movies.drop(columns=["movieId", "title"]).values
y = movies["movieId"].values

# Define the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation="relu", input_shape=(x.shape[1],)),
  tf.keras.layers.Dense(16, activation="relu"),
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss="mean_squared_error", optimizer="adam")

# Train the model
model.fit(x, y, epochs=10, batch_size=32)

# Use the model to predict the movieId of the best movie
best_movie_id = model.predict(x)

# Find the movie with the highest predicted movieId
best_movie = movies[movies["movieId"] == best_movie_id]

# Print the title of the best movie
print(f"Best movie: {best_movie['title'].values[0]}")
