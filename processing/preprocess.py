import os
import music21 as m21
import json
import numpy as np
import tensorflow.keras as keras

# music21 helps to work with symbolic data and helps convert data from one format to another
# first loading the songs in kern from the file
# dataset_path is the path to the songs

KERN_DATASET_PATH = "deutschl/test"
SAVE_DIR = "Dataset"
single_file_dataset = "file_dataset"
sequence_length = 64
mapping_path = "mapping.json"
# 64 items

acceptable_durations = [  # all the duration accepted in quarter not length
    0.25,  # 1/ 16th note , each time stamp is equal ton quarter length note
    0.5,  # 1/8th note
    0.75,
    1.0,  # 1/4 th note
    1.5,
    2,  # 1/2 note
    3,
    4  # 1note
]


def load_songs_in_kern(dataset_path):
    # go through all the files in dataset and load them with music 21
    songs = []
    # list for storing all the songs
    for path, subdirs, files in os.walk(dataset_path):
        # os.walk recuresively goes through all the files in the given structure
        # path is refernce to curr subfolder
        # all the subdirectiories or subfolders in path
        # files are the files in curr directory
        for file in files:
            # filtering out non kern files
            if file[-3:] == "krn":
                # only krn files will have .krn extension
                song = m21.converter.parse(os.path.join(path, file))
                # the format of the song
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    # finding if song is acceptable or not
    # we have to analyse all the notes within the song
    for note in song.flat.notesAndRests:
        # flat :flattens the whole song structure and all objects into single list
        # notes and rests:filters out all the notes which are not notes and rest
        if note.duration.quarterLength not in acceptable_durations:
            return False
        # we do this to simplify our model
    return True


def transpose(song):
    # get key from the song(in most of the song key is notated
    # key is generally stored in the first measure of the first part of the score
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]
    # item at index 4,genreally key is stored in 4th index

    # if key is not notated than we will estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")
        # music21 will analyze the song and create the key object
    # get interval for transposition

    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song
    # transposing so that we like to reduce C minor and A major and we dont have to generalize data for large
    # otherwise ot take lot of time to run the model


def encode_song(song, time_step=0.25):
    # time step is 1/16th note
    # song is given as an input and it will be converted to time series representation
    # p =60 d=1.0 ->[60,"_" ,"_","_"] each item will correspond to 16th note
    # the integer will represent midi notes
    # r represents rest
    # '-' represents notes and rest carried forward
    encoded_song = []

    for event in song.flat.notesAndRests:
        # flattening heirachial structure of song in list
        # handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        if isinstance(event, m21.note.Rest):
            symbol = "r"

        # convert note and rest to time series notation

        steps = int(event.duration.quarterLength / time_step)
        # type conversion for the for loop
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")
    # cast encoded song to a str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):
    pass
    # load the folks songs
    print("The song is being loaded")
    songs = load_songs_in_kern(dataset_path)
    print(f"Files is loaded {len(songs)}")
    for i, song in enumerate(songs):
        # filter out songs that have non-acceptable durations
        if not has_acceptable_durations(song, acceptable_durations):
            continue
            # song is skipped
        # Transpose songs to Cmajor/Amin
        song = transpose(song)
        # encode the songs with music time series representaion
        encoded_song = encode_song(song)
        # save the songs to text file which are recommended

        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_Single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    # load the enoded songs and add delimeters
    # delimeter here is sequence length so that two songs can be seperated
    new_song_delimeter = "/ " * sequence_length
    songs = ""
    # updated for every song
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            # append song to our strings
            songs = songs + song + " " + new_song_delimeter
    songs = songs[:-1]
    # slicing or removing the last extra whitespace
    # save string that contains all dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)
    return songs


def create_mapping(songs, mapping_path):
    # mapping all symbols to integer
    # we will try to map every values
    mappings = {}
    # identify vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))
    # casting songs to a list

    # create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i
    # save vocabulary to json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    # covert song in symbolic representaition to integer into sequences so that we can train our LSTM model
    # takes input as songs which is string of integer
    int_songs = []
    # store all dataset and mapping
    # load the mappings
    with open(mapping_path, "r") as fp:
        mappings = json.load(fp)
        # dictionary which contain all our symbols

    # cast songs string to list
    songs = songs.split()
    # splits a string at empty spaces and create items in list

    # map song to int
    for symbol in songs:
        int_songs.append(mappings[symbol])
    #print(int_songs)
    return int_songs


def generating_training_sequences(sequence_length):
    # load the songs and map to int
    songs = load(single_file_dataset)
    #print(songs)
    int_songs = convert_songs_to_int(songs)
    # genearte the training sequence
    num_sequences=len(int_songs)-sequence_length
    input=[]
    target=[]
    for i in range(num_sequences):
        input.append(int_songs[i:i+sequence_length])
        target.append(int_songs[i+sequence_length])
    # one-hot encode the sequences
    # inputs: (# of sequences ,sequence length,vocabulary length)
    # easiest way to work with categorical data for neural network
    vocabulary_size=len(set(int_songs))
    inputs = keras.utils.to_categorical(input, num_classes=vocabulary_size)
    # input will now be a 3D array
    targets=np.array(target)
    return inputs,targets

def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_Single_file_dataset(SAVE_DIR, single_file_dataset, sequence_length)
    create_mapping(songs, mapping_path)
    inputs,targets=generating_training_sequences(sequence_length)

if __name__ == "__main__":
    main()
