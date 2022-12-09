import tensorflow as tf
import pandas as pd
""" mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test) """


genome_scores = pd.read_csv(
    "genome-scores.csv",
    names=["movieId", "tagId", "relevance"])

genome_scores.head()


genome_tags = pd.read_csv(
    "genome-tags.csv",
    names=["tagId", "tag"])

genome_tags.head()


links = pd.read_csv(
    "links.csv",
    names=["movieId", "imdbId", "tmdbId"])

links.head()

movies = pd.read_csv(
    "movies.csv",
    names=["movieId", "title", "genres"])

movies.head()


ratings = pd.read_csv(
    "ratings.csv",
    names=["userId", "movieId", "rating", "timestamp"])

ratings.head()


tags = pd.read_csv(
    "tags.csv",
    names=["userId", "movieId", "tag", "timestamp"])

tags.head()

