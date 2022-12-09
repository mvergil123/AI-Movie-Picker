import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
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
    names=["tagId", "tag"], skiprows=1)

genome_tags.head()

genome_tags_features = genome_tags.copy()
genome_tags_labels = genome_tags_features.pop("tagId")
print(genome_tags_labels)
# input("Check this!!! ")
genome_tags_features = np.array(genome_tags_features)

genome_tags_model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

genome_tags_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

genome_tags_model.fit(genome_tags_features, genome_tags_labels, epochs=10)

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

