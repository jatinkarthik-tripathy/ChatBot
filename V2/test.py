from seq2seq import seq2seq
import pickle
import numpy as np


[convs_from, convs_to, input_token, output_token, NUM_ENC_TOKENS, NUM_DEC_TOKENS,
 MAX_ENC_LEN, MAX_DEC_LEN] = pickle.load(open('parameters.pkl', 'rb'))
s2s = seq2seq()
s2s.create_models(NUM_ENC_TOKENS, NUM_DEC_TOKENS)

s2s.load_model("seq2seq_model.h5")


while True:
    text = input('>>')
    if text == "stop":
        quit()
    encoder_input_data = np.zeros(
        (1, MAX_ENC_LEN, NUM_ENC_TOKENS), dtype='float32')
    for t, char in enumerate(text):
        encoder_input_data[0, t, input_token[char]] = 1.
    op_sentence = s2s.test(encoder_input_data, input_token,
                           output_token, NUM_DEC_TOKENS, MAX_DEC_LEN)
    print(op_sentence)
