import json
import numpy as np 
from seq2seq_attn import seq2seq
import pickle

def get_tokens():
    with open('data/input_token_5_wrds.json') as f:
        inp_vocab = json.load(f)
    with open('data/output_token_5_wrds.json') as f:
        out_vocab = json.load(f)
    
    return inp_vocab, out_vocab

def get_convs():
    ques = []
    ans = []
    with open('data/clean_conversations_5_wrds.json') as f:
        convs = json.load(f)
    for q, a in convs.items():
        q = q.replace('.', '').replace('*', '').replace('\t', '')
        a = a.replace('.', '').replace('*', '').replace('\t', '')
        a = '\t' + a + '\n'
        ques.append(q)
        ans.append(a)
    
    return ques, ans


input_token, output_token = get_tokens()
convs_from, convs_to = get_convs()
print(convs_from[0:3])
print(convs_to[0:3])
MAX_ENC_LEN = max([len(txt) for txt in convs_from])
MAX_DEC_LEN = max([len(txt) for txt in convs_to])
NUM_ENC_UNITS = len(input_token)
NUM_DEC_UNITS = len(output_token)
print(f'input:{NUM_ENC_UNITS}')
print(f'output:{NUM_DEC_UNITS}')
print(f'max enc:{MAX_ENC_LEN}')
print(f'max dec:{MAX_DEC_LEN}')
print(f'input sentences:{len(convs_from)}')
print(f'output sentences:{len(convs_to)}')
print(len(convs_from) * MAX_ENC_LEN * NUM_ENC_UNITS*4)
encoder_input_data = np.zeros(
    (len(convs_from), MAX_ENC_LEN, NUM_ENC_UNITS),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(convs_from), MAX_DEC_LEN, NUM_DEC_UNITS),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(convs_from), MAX_DEC_LEN, NUM_DEC_UNITS),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(convs_from, convs_to)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token[char]] = 1.
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, output_token[char]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, output_token[char]] = 1.
parameters = [convs_from, convs_to, input_token, output_token, NUM_ENC_UNITS, NUM_DEC_UNITS,
              MAX_ENC_LEN, MAX_DEC_LEN]
pickle.dump(parameters, open('parameters.pkl', 'wb'))

s2s = seq2seq()
s2s.create_models(NUM_DEC_UNITS, NUM_DEC_UNITS)
s2s.train()
