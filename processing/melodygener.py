import json
import tensorflow.keras as keras
import music21 as m21
from preprocess import sequence_length, mapping_path
import numpy as np
SAVE_DIR = "Dataset"

class MelodyGenerator:

    def __init__(self, model_path="model.h5"):
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        # loaded the keras model
        # sequence length,look up table or mapping

        with open(mapping_path, "r") as fp:
            self._mappings = json.load(fp)
            # load the mappings in _mappings

        self._start_symbols = ["/"] * sequence_length

    def generate_melody(self, seed, num_steps, max_sequence_length, temp):
        # "64 _ 63 _ _ " melody encoded in time series
        # seed is a piece of melody for network
        # num_steps :number of steps in time series representation in which our output will be generated,
        # max_sequence_length: is the maximum seed length
        # temp is a flot with range 0,1

        # create seed with start symbol
        seed = seed.split()
        # melody will store all symbols produced by network
        melody = seed
        seed = self._start_symbols + seed
        # map seed to integers
        seed = [self._mappings[symbol] for symbol in seed]

        for i in range(num_steps):

            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            # one-hot encode seed
            one_hot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # one_hot_seed:an array of 2D size will be equal to (max_sequnce_length,len(mappings))
            one_hot_seed = one_hot_seed[np.newaxis, ...]
            # adds an extra dimension which will be required to predict in keras
            # one_hot_seed :will now be a 3D array

            # make predictions
            probablity = self.model.predict(one_hot_seed)[0]

            output_int = self._sample_with_temp(probablity, temp)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # check whether we are at end of melody
            if output_symbol == "/":
                break
            # update the melody
            melody.append(output_symbol)
        return melody

    def _sample_with_temp(self, probablity, temp):
        # temp is infinity probablity will become homogenous as if picking any random indexes
        # temp is close to zero ,proabality will become deterministic
        # temp is equal to 1 than it will be same as original
        predictions = np.log(probablity) / temp
        proabablity = np.exp(predictions) / np.sum(np.exp(predictions))  # softmax function
        # if temp is greater than 1 our "predictions" will lie in smaller range and probablity will have homogenous
        # distribution if temp is small than predicitons will be large and after applying softmax function than high
        # probablity index will have higher chance to pick

        choices = range(len(probablity))
        index = np.random.choice(choices, p=probablity)

        return index

    def _file_name(self,count):
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
    def save_melody(self, melody, step_duration=0.25, format="midi", save_file_name="mel3.midi"):

        # create a music21 stream
        stream = m21.stream.Stream()

        # parse all the symbols in melody and create note/rest objects basically music21 object
        start_symbol = None
        # It can be midi note or rest
        step_counter = 1

        for i, symbols in enumerate(melody):
            # handle case in which we have a note/rest
            if symbols != "_" or i + 1 == len(melody):
                # for the first symbol it is important so that
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    if start_symbol == "r":
                        # handle rest
                        m21_event = m21.note.Rest(quarter_length_duration)
                    else:
                        # handle note
                        m21_event = m21.note.Note(int(start_symbol), quarterlength=quarter_length_duration)
                    stream.append(m21_event)
                    step_counter = 1
                    # for next iteration

                start_symbol = symbols
            else:
                # handle case for prolongation
                step_counter += 1

        # write the m21 stream to midi file
        stream.write(format, save_file_name)


if __name__ == '__main__':
    mg = MelodyGenerator()
    seed = "55 _ 60 _ 60 _ 60 _ 64 _ 67 _ 64 _"
    seed2 = "55 60 _ 60 _ 60 _ 64 _ 67 _ 64 _ 60 _ _ _ 64 _ 67 _ 67 _ 67 _ 72 _ 67 "
    # melody = mg.generate_melody(seed, 500, sequence_length, 0.5)
    melody1 = mg.generate_melody(seed2, 500, sequence_length, 0.5)
    # print(melody)
    # mg.save_melody(melody)
    mg.save_melody(melody1)
