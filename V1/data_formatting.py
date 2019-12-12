import json
from colorama import Fore, Style, init
import pickle
init()


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def vocab_build():
	#read json file
	with open('data/clean/conversations.json', 'r') as f:
	    convs = json.load(f)

	ques = []
	ans = []
	input_vocab = []
	output_vocab = []
	counter = 0
	tot = len(convs)
	print(f'Total conversations: {tot}')
	printProgressBar(0, tot, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for q, a in convs.items():
		q = q.strip().replace('.', '').replace('\t', '').replace('*', '')
		a = a.strip().replace('.', '').replace('\t', '').replace('*', '')
		a = '\t' + a + '\n'
		ques.append(q)
		ans.append(a)

		for char in q:
			if char not in input_vocab:
				input_vocab.append(char)
		for char in a:
			if char not in output_vocab:
				output_vocab.append(char)
		counter += 1
		printProgressBar(counter, tot, prefix = 'Progress:', suffix = 'Complete', length = 50)

	input_vocab = sorted(input_vocab)
	input_vocab = dict([(char, i) for i, char in enumerate(input_vocab)])
	with open('data/input_token.json', 'w', encoding='utf8') as f:
		json.dump(input_vocab, f, indent=4)

	output_vocab = sorted(output_vocab)
	output_vocab = dict([(char, i) for i, char in enumerate(output_vocab)])
	with open('data/output_token.json', 'w', encoding='utf8') as f:
		json.dump(output_vocab, f, indent=4)

	pickle.dump(ques, open('data/convs_from.pkl', 'wb'))
	pickle.dump(ans, open('data/convs_to.pkl', 'wb'))

if __name__ == '__main__':
	try:
		ques = pickle.load(open('convs_from.pkl', 'rb'))
		ans = pickle.load(open('convs_to.pkl', 'rb'))

		with open('data/input_token.json', 'r', encoding='utf8') as f:
			pass
		with open('data/output_token.json', 'r', encoding='utf8') as f:
			pass
	except FileNotFoundError:
		vocab_build()
	print(f'{Fore.GREEN}Data Formatting Complete{Style.RESET_ALL}')
