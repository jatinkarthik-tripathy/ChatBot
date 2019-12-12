import json
from colorama import Fore, Style, init
init()

def create_dict():
	id_dict = {}
	with open('data/raw/movie_lines.txt', 'r') as f:
		lines = f.read().split('\n')

	for line in lines:
		line_part = line.split(' +++$+++ ')
		if len(line_part) == 5:
			id_dict[line_part[0]] = line_part[4]

	with open('data/clean/id_dict.json', 'w') as f:
		json.dump(id_dict, f, sort_keys=True, indent=4)

	return id_dict

def create_convo(id_dict):
	convs = []
	convo_dict = {}
	with open('data/raw/movie_conversations.txt', 'r') as f:
		data = f.read().split('\n')
	for convo in data:
		convo_part = convo.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
		convs.append(convo_part.split(','))

	for convo in convs:
		if len(convo)%2 != 0:
			convo = convo[:-1]
		for i in range(len(convo)):
			if i%2 == 0:
				ques = id_dict[convo[i]]
			else:
				convo_dict[ques] = id_dict[convo[i]]

	with open('data/clean/conversations.json', 'w') as f:
		json.dump(convo_dict, f, indent=4)

if __name__ == '__main__':
	try:
		with open('data/clean/id_dict.json', 'r') as f:
			id_dict = json.load(f)
	except FileNotFoundError:
		id_dict = create_dict()
	try:
		with open('data/clean/conversations.json', 'r') as f:
			pass
	except FileNotFoundError:
		create_convo(id_dict)
	print(f'{Fore.GREEN}Processing Complete{Style.RESET_ALL}')
