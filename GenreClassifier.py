# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:19:38 2022

Genre Classifier

PLEASE NOTE:
    Although it's possible to do, so this code is not meant 
    to be run all at once, but rather to be used as a reference.
    It is advised to split the code in chunks and run
    them one after another in a notebook.
"""

# Necessary libraries

import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

####################################################################
# LOADING, FORMATTING, AND SPLITTING THE DATA
####################################################################

# Prepares the song by producing spectograms
def prepare_song(song_path):
  list_matrices = []
  y, sr = librosa.load(song_path,sr=None)
  melspect = librosa.feature.melspectrogram(y)
  list_matrices.append(melspect[:,:1290])
  return list_matrices

# all tracks will be the X features and genre will be the target y
all_tracks = []
genre = []

# Defines desired genre and loads the tracks, with the correct
# labes into the variables all_tracks and genre.
genres=['classical','metal','blues','country',
        'disco','hiphop','pop','reggae','rock']

y_output=np.identity(len(genres))

# Use the appriopriate directory here.
dir='/content/drive/MyDrive/GenreProject_tracks/genres_original/'

for genre_name,y in list(zip(genres,y_output)):
  for song in os.listdir(dir+genre_name):
    try:
      song_pieces=prepare_song(dir+genre_name+'/'+song)
      all_tracks+=song_pieces
      genre+=(list(y)*len(song_pieces))
    except:
      continue
all_tracks=np.array(all_tracks)

# Scales the spectograms to lie between 0 and 1.
alt=np.array(all_tracks)
all_tracks_scaled=(alt-np.min(alt))/(np.max(alt)-np.min(alt))
genre=np.array(genre)
genre=np.reshape(genre,(-1,len(genres)))

# Checks that the input and labes have the desired shapes.
print(np.shape(all_tracks_scaled))
print(np.shape(genre))

#Splits the data into test, validation, and training.
X1_train, X1_test, y1_train, y1_test = train_test_split(all_tracks_scaled, 
                                                    genre,
                                                    train_size=0.8,
                                                    random_state=40,
                                                    stratify=genre)

X1_val, X1_test, y1_val, y1_test = train_test_split(X1_test, 
                                                y1_test,
                                                test_size=0.5,
                                                stratify=y1_test,
                                                random_state=40)

# Defines functions to split each song and its corresponding label in 10.
def multiply_y(genre_output):
  multiplied=[]
  for y in genre_output:
    multiplied.append(list(y)*10)
  
  multiplied=np.array(multiplied)
  multiplied=np.reshape(multiplied,(-1,len(genres)))
  return multiplied

def split_song(data):
  data_split=np.split(data[0],10,axis=1)
  for song in data[1:]:
    song_pieces=np.split(song,10,axis=1)
    data_split=np.vstack((data_split,song_pieces))

  return data_split
  
# Splits and shuffles the data
X_train=split_song(X1_train)
X_val=split_song(X1_val)
X_test=split_song(X1_test)

y_train=multiply_y(y1_train)
y_val=multiply_y(y1_val)
y_test=multiply_y(y1_test)

indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)

X_train=X_train[indices]
y_train=y_train[indices]

indices = np.arange(X_val.shape[0])
np.random.shuffle(indices)

X_val=X_val[indices]
y_val=y_val[indices]

indices = np.arange(X_test.shape[0])
np.random.shuffle(indices)

X_test=X_test[indices]
y_test=y_test[indices]

####################################################################
# TRAINING THE NETWORK
####################################################################

# Defines the neural network.

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128,129, 1)))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))

model.add(layers.Dense(len(genres), activation='softmax'))

# Training the model.
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0002),
              metrics='accuracy')

history = model.fit(X_train, y_train, batch_size=10, epochs=40, 
                    validation_data=(X_val, y_val))

# Plots the training and validation accuracy.
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluates the model using the test data.
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)

####################################################################
# PRODUCING CONFUSION MATRIX
####################################################################

# Generates the confusion matrix.
size=len(genres)
matrix = np.zeros((size,size))

# Calculates the number of tracks from the different genres in the test set.
size=len(genres)
total_genres=[]
for i in range(size):
    total_genres.append(np.sum(y_test[:,i]))
print(total_genres)

# Evaluates the prediction on each track, and inputs the prediction
# in the appropriate part of the matrix.
for i in range(len(y_test)):
    correct=np.where(y_test[i]==1)
    prediction=model.predict(X_test[i:i+1])
    guess=np.argmax(prediction, axis=1)
    matrix[correct,guess]+=1

# Gets the ratio of guesses wrt. the number of tracks.
for i,number in enumerate(total_genres):
    matrix[i]=matrix[i]/number

# Sort by diagonal.
diag = np.diag(matrix)
idx = np.argsort(diag)
idx = idx[::-1]
total_genres=total_genres[idx]
matrix = matrix[idx,:][:,idx]
mylist = [genres[i] for i in idx]

# Computes the confidence interval for the ratio of correct guesses.
nymatrix = matrix.astype(object)
for k,i in enumerate(np.diag(matrix)):
   confidence = 1.96 * np.sqrt((i*(1-i))/total_genres[k])
   confidence = round(confidence,2)
   interval = "{} ± {}".format(i,confidence)
   nymatrix[k,k] = interval

# Plots the confusion matrix.
plt.figure(dpi=250)
fig, ax = plt.subplots(figsize=(6,6))

# Set the colourmap of the diagonal to white-blue, and the rest 
# to white-red.
off_diag_mask = np.eye(*matrix.shape, dtype=bool)
diag_mask = np.invert(off_diag_mask)
maskedmatrix = np.ma.masked_where(off_diag_mask == True ,matrix)
unmaskedmatrix = np.ma.masked_where(off_diag_mask == False ,matrix)

ax.matshow(unmaskedmatrix, cmap=plt.cm.Blues, vmin = 0.5, vmax=1.2)
ax.matshow(maskedmatrix,cmap = plt.cm.Oranges, vmin = 0, vmax = 0.5)

# Insert the ratios in the appropriate spot of the matrix.
for i in range(size):
    for j in range(size):
        text = nymatrix[j,i]
        ax.text(i, j, str(text), va='center', ha='center')

ax.set_title('Confusion matrix', pad=20)
ax.set_xlabel('Guesses')
ax.xaxis.set_label_position('top') 
ax.set_ylabel('Correct genres')
plt.yticks(range(size),mylist)
plt.xticks(range(size),mylist)
plt.show()

####################################################################
# TESTING ON SONGS
####################################################################

# Define path to song (adjust to your environment).
example_directory='/content/drive/MyDrive/GenreProject_tracks/Examples/'
song='jolene.wav'
song_path=example_directory+song

# Extract the spectogram for the song and split it into roughly 3-second bites.
def sample_song(song_path):
  y,sr = librosa.load(example_directory+song,sr=None)
  melspect = librosa.feature.melspectrogram(y)
  melspect=(melspect-np.min(melspect))/(np.max(melspect)-np.min(melspect))

  samples=int(np.floor(len(melspect[0])/129))
  melspect=melspect[:,:int(samples*129)]
  sample_list=np.split(melspect,samples,axis=1)
  sample_list=np.expand_dims(sample_list, axis=3)

  return sample_list

# Readies the data for evaluation by the model.
sample_list=sample_song(song_path)
overall_prediction=np.zeros(len(genres))
overall_prediction=np.expand_dims(overall_prediction, axis=0)

# Each 3-second chunk gets evaluated by the model.
for i in range(len(sample_list)):
  prediction=model.predict(sample_list[i:i+1])
  overall_prediction[:,np.argmax(prediction)]+=1

# The average of the predictions is outputted, along with the genre
# with the highest value.
guess=np.argmax(overall_prediction)
overall_prediction=np.squeeze(overall_prediction)
overall_prediction=overall_prediction/len(sample_list)
genre_guess=genres[guess]

# The prediction of the model is illustrated with a bar plot.
fig, ax = plt.subplots()
ax.bar(genres, overall_prediction, width=0.7, color='steelblue')
ax.set_title('Genre classification \n (Jolene - Dolly Parton)', pad=20)
ax.set_xlabel('Genres') 
ax.set_ylabel('Confidence')
ax.set_ylim(0,1)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)

plt.margins(0.1)
plt.show()
