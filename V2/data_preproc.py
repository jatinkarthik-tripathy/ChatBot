import json


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
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
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def shorten(part, size):
    """
    Call in a loop to shorten the sentence
    @params:
        part - Required : sentence that need to be shortened (Str)
    """
    parts = part.split(" ")
    if len(parts) > size:
        if len(part.split(".")) > 1:
            part = part.split(".")[0]
        else:
            part = ""
            for bit in parts[0:(size-1)]:
                part += bit + ' '
    # print(part)
    return part


if __name__ == "__main__":
    id_dict = {}
    with open('data/movie_lines.txt', 'r') as f:
        lines = f.read().split('\n')

    for line in lines:
        line_part = line.split(' +++$+++ ')
        if len(line_part) == 5:
            id_dict[line_part[0]] = line_part[4]

    convs = []
    convo_dict = {}
    with open('data/movie_conversations.txt', 'r') as f:
        data = f.read().split('\n')
    for convo in data:
        convo_part = convo.split(
            ' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
        convs.append(convo_part.split(','))

    tot_convs = len(convs)
    counter = 0
    printProgressBar(counter, tot_convs, prefix='Progress:',
                     suffix='Complete', length=50)
    for convo in convs:
        if len(convo) % 2 != 0:
            convo = convo[:-1]
        for i in range(len(convo)):
            if i % 2 == 0:
                ques = id_dict[convo[i]]
                while len(ques.split(" ")) > 5:
                    ques = shorten(ques, 5)

            else:
                ans = id_dict[convo[i]]
                while len(ans.split(" ")) > 5:
                    ans = shorten(ans, 5)
                convo_dict[ques] = ans
        counter += 1
        printProgressBar(counter, tot_convs, prefix='Progress:',
                         suffix='Complete', length=50)

    with open('data/clean_conversations_5_wrds.json', 'w') as f:
        json.dump(convo_dict, f, indent=4)
