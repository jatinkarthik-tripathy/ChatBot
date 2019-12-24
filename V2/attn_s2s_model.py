import tensorflow as tf
import numpy as np
from attention_layer import AttentionLayer


class seq2seq:
    def create_models(self, NUM_ENC_TOKENS, NUM_DEC_TOKENS, MAX_ENC_LEN, MAX_DEC_LEN, latent_dim=256):
        self.NUM_ENC_TOKENS = NUM_ENC_TOKENS
        self.NUM_DEC_TOKENS = NUM_DEC_TOKENS
        self.MAX_ENC_LEN = MAX_ENC_LEN
        self.MAX_DEC_LEN = MAX_DEC_LEN
        self.latent_dim = latent_dim

        encoder_inputs = tf.keras.layers.Input(
            shape=(self.MAX_ENC_LEN, self.NUM_ENC_TOKENS), name='encoder_inputs')

        decoder_inputs = tf.keras.layers.Input(
            shape=(self.MAX_DEC_LEN, self.NUM_DEC_TOKENS), name='decoder_inputs')
        # Encoder GRU
        encoder_gru = tf.keras.layers.GRU(self.latent_dim, return_sequences=True,
                                          return_state=True, name='encoder_gru')
        encoder_out, encoder_state = encoder_gru(encoder_inputs)

        # Set up the decoder GRU, using `encoder_states` as initial state.
        decoder_gru = tf.keras.layers.GRU(self.latent_dim, return_sequences=True,
                                          return_state=True, name='decoder_gru')
        decoder_out, decoder_state = decoder_gru(
            decoder_inputs, initial_state=encoder_state)

        # attention
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_out, decoder_out])

        decoder_concat_input = tf.keras.layers.Concatenate(
            axis=-1, name='concat_layer')([decoder_out, attn_out])

        # Dense layer
        dense = tf.keras.layers.Dense(
            self.NUM_DEC_TOKENS, activation='softmax', name='softmax_layer')
        dense_time = tf.keras.layers.TimeDistributed(
            dense, name='time_distributed_layer')
        decoder_pred = dense_time(decoder_concat_input)

        self.train_model = tf.keras.models.Model(
            [encoder_inputs, decoder_inputs], decoder_pred)
        self.train_model.compile(
            optimizer='adam', loss='categorical_crossentropy')
        print(self.train_model.summary())

        # Inference setup:
        inf_encoder_inputs = tf.keras.layers.Input(
            shape=(self.MAX_ENC_LEN, self.NUM_ENC_TOKENS), name='encoder_inputs')
        inf_enc_out, inf_enc_state = encoder_gru(inf_encoder_inputs)
        self.encoder_model = tf.keras.models.Model(inputs=inf_encoder_inputs, outputs=[
            inf_enc_out, inf_enc_state])

        print(self.encoder_model.summary())

        decoder_inf_inputs = tf.keras.layers.Input(
            shape=(1, 1, self.NUM_DEC_TOKENS), name='decoder_inputs')
        encoder_inf_states = tf.keras.layers.Input(shape=(
            1, self.MAX_ENC_LEN, self.latent_dim), name='encoder_inf_states')
        decoder_init_state = tf.keras.layers.Input(shape=(
            None, self.latent_dim), name='decoder_init')

        decoder_inf_out, decoder_inf_state = decoder_gru(
            decoder_inf_inputs, initial_state=decoder_init_state)
        attn_inf_out, attn_inf_states = attn_layer(
            [encoder_inf_states, decoder_inf_out])
        decoder_inf_concat = tf.keras.layers.Concatenate(
            axis=-1, name='concat')([decoder_inf_out, attn_inf_out])
        decoder_inf_pred = tf.keras.layers.TimeDistributed(
            dense)(decoder_inf_concat)
        self.decoder_model = tf.keras.models.Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                                              outputs=[decoder_inf_pred, attn_inf_states, decoder_inf_state])
        print(self.decoder_model.summary())

    def load_model(self, name):
        self.train_model.load_weights(name)

    def train(self, enc_in, dec_in, dec_out, batch=100, epochs=20, cbs=[]):
        self.train_model.fit([enc_in, dec_in], dec_out,
                             batch_size=batch, epochs=epochs, callbacks=cbs)
        self.train_model.save('attn_5words.h5')

    def test(self, input_seq, input_token_index, target_token_index, num_decoder_tokens, max_decoder_seq_length):
        # Reverse-lookup token index to decode sequences back to
        # something readable.
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

        # Encode the input as state vectors.
        enc_outs, enc_last_state = self.encoder_model.predict(input_seq)
        dec_state = enc_last_state
        attention_weights = []
        
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        print(target_seq.shape)
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            dec_out, attention, dec_state = self.decoder_model.predict(
                [enc_outs, dec_state, target_seq])
                
            # Sample a token
            sampled_token_index = np.argmax(dec_out, axis=-1)[0, 0]
            sampled_token_index = np.argmax(dec_out[0, -1, :])
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
            attention_weights.append((dec_ind, attention))

        return decoded_sentence
