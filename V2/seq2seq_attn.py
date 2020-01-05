import tensorflow as tf
import numpy as np


class seq2seq:
    def attention(self, enc_out, hidden):
        self.W1 = tf.keras.layers.Dense(self.NUM_DEC_UNITS)
        self.W2 = tf.keras.layers.Dense(self.NUM_DEC_UNITS)
        self.V = tf.keras.layers.Dense(1)

        hidden_expanded = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W1(enc_out) +
                                  self.W2(hidden_expanded)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * enc_out
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def create_models(self, NUM_ENC_UNITS, NUM_DEC_UNITS, latent_dim=256, batch_size=250):
        self.NUM_ENC_UNITS = NUM_ENC_UNITS
        self.NUM_DEC_UNITS = NUM_DEC_UNITS
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        # training model:
        # encoder model
        encoder_inputs = tf.keras.layers.Embedding(self.NUM_ENC_UNITS, self.latent_dim)
        encoder = tf.keras.layers.CuDNNGRU(
            self.NUM_ENC_UNITS, return_sequences=True,
            return_state=True)
        enc_outputs, enc_hidden = encoder(encoder_inputs)

        # decoder model
        context_vector = self.attention(enc_outputs, enc_hidden)
        target_inputs = tf.keras.layers.Embedding(self.NUM_DEC_UNITS, self.latent_dim)
        concat = tf.concat(
            [tf.expand_dims(context_vector, 1), target_inputs], axis=-1)
        decoder_inputs = tf.keras.layers.Input(shape=concat.shape)
        decoder_gru = tf.keras.layers.CuDNNGRU(
            self.NUM_DEC_UNITS, return_sequences=True,
            return_state=True)
        decoder_outputs, _, = decoder_gru(decoder_inputs)
        decoder_outputs = tf.reshape(decoder_outputs, (-1, decoder_outputs.shape[2]))
        decoder_dense = tf.keras.layers.Dense(
            self.NUM_DEC_UNITS, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        self.train_model = tf.keras.models.Model(
            [encoder_inputs, decoder_inputs], decoder_outputs)
        self.train_model.compile(
            optimizer='rmsprop', loss='categorical_crossentropy')

        # Inference setup:
        self.encoder_model = tf.keras.models.Model(
            encoder_inputs, enc_hidden)

        decoder_state_input_h = tf.keras.layers.Input(shape=(self.latent_dim,))
        decoder_state_input_c = tf.keras.layers.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_gru(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = tf.keras.models.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    def load_model(self, name):
        self.train_model.load_weights(name)

    def train(self, enc_in, dec_in, dec_out, epochs=20):
        self.train_model.fit([enc_in, dec_in], dec_out,
                             batch_size=self.batch_size, epochs=epochs)
        self.train_model.save('seq2seq_model.h5')

    def test(self, input_seq, input_token_index, target_token_index, num_decoder_tokens, max_decoder_seq_length):
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence
