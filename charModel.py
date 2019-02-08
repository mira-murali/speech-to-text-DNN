import numpy as np
import os, sys
import argparse
import csv

parser = argparse.ArgumentParser('Character-based Model for LAS')
parser.add_argument('--data-path', default='all', type=str, help='Path to data')
parser.add_argument('--write-file', default='', type=str, help='csv file to write vocabulary')
parser.add_argument('--trans-file', default='', type=str, help='new transcript (npy) file to write to')
parser.add_argument('--first', default=1, type=int, help='if vocab.csv needs to be generated')

args = parser.parse_args()
orig_transcripts = np.load(args.data_path)
vocab = {'SOS': 0, 'EOS':1, ' ':2}
count = 3
new_word = []
new_transcript = []
index = 0
for example in orig_transcripts:
    new_transcript.append([])
    new_transcript[index].append(int(vocab['SOS']))
    for word in example:
        decoded_word = word.decode('utf-8')
        for l in range(len(decoded_word)):
            if decoded_word[l] not in vocab.keys():
                vocab[decoded_word[l]] = count
                count += 1

            new_transcript[index].append(int(vocab[decoded_word[l]]))
        new_transcript[index].append(int(vocab[' ']))
    new_transcript[index].append(int(vocab['EOS']))
    new_transcript[index] = np.array(new_transcript[index])
    index += 1

np.save(os.path.join('', args.trans_file), np.array(new_transcript))
if args.first:
    with open(os.path.join('', args.write_file), 'w') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in vocab.items():
            writer.writerow([key, value])
