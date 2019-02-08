import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import csv
import argparse
import time
import math
import Levenshtein as L


parser = argparse.ArgumentParser('Training LAS')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs to train model')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate for optimizer')
parser.add_argument('--foldername', default='baseline', type=str, help='folder to store model in')
parser.add_argument('--sub-name', default='submission.csv', type=str, help='csv file name to save test submission')
parser.add_argument('--batch-size', default=32, type=int, help='batch size for the input')
parser.add_argument('--data-path', default='all', type=str, help='path to folder containing data')
parser.add_argument('--resume', default=0, type=int, help='set to 1 if loading in trained model')
parser.add_argument('--eval', default=0, type=int, help='set to 1 if loading in saved model just to test')
parser.add_argument('--finetune', default=0, type=int, help='set to 1 to load in trained model and train further')
parser.add_argument('--load-folder', default='', type=str, help='path to saved model if eval/resume/finetune is 1')
parser.add_argument('--print-freq', default=50, type=int, help='how often performance must be printed per epoch')
parser.add_argument('--pretrain', default='', type=str, help='path to load in pretrained model for speller')
args = parser.parse_args()
dir_path = os.path.realpath('las.py')
os.environ['CURRENT'] = dir_path[:dir_path.find('las')]
if not os.path.isdir(os.path.join(os.environ['CURRENT'], args.foldername)):
    os.mkdir(os.path.join(os.environ['CURRENT'], args.foldername))

with open(os.path.join(args.data_path, 'vocab.csv'), 'r') as csv_file:
    reader = csv.reader(csv_file)
    vocab = dict(reader)

# also getting the reverse
devocab = dict((int(v), k) for k, v in vocab.items())

if args.eval:
    outfile = open(os.path.join(args.foldername, 'output_eval.txt'), 'w')
else:
    outfile = open(os.path.join(args.foldername, 'output.txt'), 'w')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
outfile.write(str(device))
outfile.write('\n')

def get_dist(predictions, labels, label_lengths):
    ls = 0
    for i in range(predictions.size(0)):
            pred = "".join(devocab[int(o)] for o in predictions[i, :label_lengths[i]])
            true = "".join(devocab[int(l)] for l in labels[i])
            # print("Pred: {}, True: {}".format(pred, true))
            ls += L.distance(pred, true)
        # assert pos == labels.size(0)
    return ls / predictions.size(0)

def plot_attention(attention, name='attention.png'):
    # attention is LD x LE where LD is decoder time steps and LE is encoder time steps
    plt.imshow(attention.detach().cpu().numpy()[0])
    # plt.show()
    plt.savefig(os.path.join(args.foldername, name))

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if (epoch+1)>=int(args.epochs)//2:
        lr = lr*0.5
    elif (epoch+1)>=(0.67*args.epochs):
        lr = lr*0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def write_to_csv(predictions):
    output_file = open(os.path.join(args.foldername, args.sub_name), 'w')
    pred_writer = csv.writer(output_file, delimiter=',')
    pred_writer.writerow(['Id', 'Predicted'])
    for i in range(len(predictions)):
        sentence = predictions[i]
        pred_writer.writerow([str(i), sentence])

class framesDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = [torch.tensor(frames[i]) for i in range(frames.shape[0])]
        if labels is not None:
            self.labels = [torch.LongTensor(labels[i]) for i in range(labels.shape[0])]
        else:
            self.labels = None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.labels is not None:
            input = self.labels[idx][:-1]
            target = self.labels[idx][1:]
            return frame.to(device), input.to(device), target.to(device)
        return frame.to(device)

def collate_frames(utterance_list):
    utterances, inputs, targets = zip(*utterance_list)
    seq_lengths = [len(utterance) for utterance in utterances]
    seq_order = sorted(range(len(seq_lengths)), key=seq_lengths.__getitem__, reverse=True)
    sorted_utterances = [utterances[i] for i in seq_order]
    sorted_targets = [targets[i] for i in seq_order]
    sorted_inputs = [inputs[i] for i in seq_order]
    return sorted_utterances, sorted_inputs, sorted_targets

class Encoder(nn.Module):
    def __init__(self, hidden_size=256, input_size=40, nlayers=4, out_features=128):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = out_features
        self.nlayers = nlayers
        self.rnns = nn.ModuleList()
        self.rnns.append(nn.LSTM(input_size = self.input_size, hidden_size=self.hidden_size, bidirectional=True))
        for i in range(self.nlayers-2):
            self.rnns.append(nn.LSTM(input_size = self.hidden_size*2, hidden_size=self.hidden_size, bidirectional=True))
        self.rnns.append(nn.LSTM(input_size=self.hidden_size*2, hidden_size=self.hidden_size, bidirectional=True))
        self.key_layer = nn.Linear(self.hidden_size*2, self.output_size)
        self.value_layer = nn.Linear(self.hidden_size*2, self.output_size)
        self.h = nn.Parameter(torch.zeros((2, 1, self.hidden_size)), requires_grad=True).to(device)

    def forward(self, utterance_list):
        # utterance_list is a list of utterances in descending order, each of dimension sequence_length x features
        batch_size = len(utterance_list)
        packed_input = pack_sequence(utterance_list)
        h = self.h.repeat((1, batch_size, 1))
        hidden = (h, h)
        output_packed, hidden_state = self.rnns[0](packed_input, hidden)
        packed_padded_input = None
        for i in range(1, self.nlayers):
            # padded_input is of shape seq_length, batch_size, features
            padded_input, seq_lens = pad_packed_sequence(output_packed)
            max_seq_len, index = seq_lens.max(0)
            # transpose to get batch_size, seq_length, features
            padded_input = padded_input.permute(1, 0, 2)
            # reshape to get batch_size, seq_length//2, features*2
            seq_length = padded_input.shape[1]
            # checking if seq_length is even
            new_seq_lengths = []

            new_padded_input = []
            for b in range(padded_input.shape[0]):
                if seq_length%2 != 0:
                    new_padded_input.append(padded_input[b, :-1, :])
                if seq_lens[b] > seq_lens[index].item()//2:
                    new_seq_lengths.append(seq_lens[0].item()//2)
                else:
                    new_seq_lengths.append(seq_lens[b].item())
            new_seq_lengths = torch.tensor(new_seq_lengths)
            if seq_length%2 != 0:
                padded_input = torch.stack(new_padded_input)
            # using contiguous to ensure there isn't a runtime error
            # padded_input = padded_input.contiguous().view(-1, padded_input.size(1)//2, 2*padded_input.size(2))
            padded_input = padded_input.contiguous().view(-1, padded_input.size(1)//2, 2, padded_input.size(2))
            # mean pooling across padded_input
            padded_input = torch.mean(padded_input, dim=2)
            # tranposing back to get original order of dimensions: seq_length//2, batch_size, features*2
            padded_input = padded_input.permute(1, 0, 2)
            # packing input to send to LSTM
            packed_padded_input = pack_padded_sequence(padded_input, new_seq_lengths)
            output_packed, hidden_state = self.rnns[i](packed_padded_input, hidden_state)

        # getting key-value pairs
        padded_input, seq_lens = pad_packed_sequence(output_packed)
        mask = torch.ones(padded_input.shape).to(device)
        padded_input_array = padded_input.clone()
        # creating a mask for the bias term in linear layer
        mask[np.where(~padded_input_array.detach().cpu().numpy().any(axis=2))] = 0
        # flattening input to send to linear layers
        padded_input = padded_input.contiguous().view(-1, padded_input.size(2))
        mask = mask.contiguous().view(-1, mask.size(2))
        key_output = self.key_layer(padded_input)
        value_output = self.value_layer(padded_input)
        mask = mask[:, :key_output.shape[1]]
        # some info about mask: linear layer is basically doing XW + B, when X has padding, W does not affect it, but adding a nonzero B does
        # so we need to replace those parts of the output with 0s again
        key_output, value_output = key_output*mask, value_output*mask
        # reshaping so we have L, B, F as the output from the linear layer
        key_output = key_output.contiguous().view(-1, batch_size, key_output.size(1))
        value_output = value_output.contiguous().view(-1, batch_size, value_output.size(1))
        return key_output, value_output

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, key_size, nlayers, vocab_size=34):
         super(Decoder, self).__init__()
         self.input_size = input_size
         self.hidden_size = hidden_size
         self.nlayers = nlayers
         self.vocab_size = vocab_size
         self.key_size = key_size
         self.embedding = nn.Embedding(self.vocab_size, self.input_size)
         self.rnncells = nn.ModuleList()
         self.rnncells.append(nn.LSTMCell(self.input_size+self.key_size, self.hidden_size))
         for i in range(nlayers-1):
             self.rnncells.append(nn.LSTMCell(self.hidden_size, self.hidden_size))


         self.projection = nn.Linear(self.hidden_size, self.key_size)
         self.scoring = nn.Linear(self.key_size*2, self.vocab_size)
         # making attention a learnable parameter
         # self.attention = nn.Parameter(torch.zeros(1, 1, 1), requires_grad = True).to(device)
         # making hidden state a learnable parameter
         self.h = nn.Parameter(torch.zeros((1, self.hidden_size)), requires_grad=True).to(device)

    def forward(self, labels, keys, values, isTest=False):
        batch_size = len(labels)
        # expand learnable hidden state parameter - now you have B x H
        h = self.h.repeat(batch_size, 1)
        # project using projection layer to get context as B x F
        context = self.projection(h)
        # unsqueeze context so you have B x 1 X F
        context = context.unsqueeze(1)
        # find the length of each transcript
        label_lengths = [len(label) for label in labels]
        # create a bounds list that will tell you where each transcript starts and ends
        bounds = [0]
        for l in label_lengths:
            bounds.append(bounds[-1]+l)
        # Now concatenate your labels to become one long tensor
        concat_labels = torch.cat(labels)
        # pass this concatenated labels tensor of shape P to your embedding layer so that you get an output P x E
        embed_output = self.embedding(concat_labels)
        # Create a list of embeddings for each of those transcripts
        embed_list = [embed_output[bounds[i]:bounds[i+1]] for i in range(batch_size)]
        # Since each of these embeddings is of variable length, pad them so they have the same sequence length M
        max_embed_length = max([len(embed) for embed in embed_list])
        # padding with negative number because label 0 exists
        padded_embeddings = [F.pad(embed, (0, 0, 0, max_embed_length - len(embed)), value=-100) for embed in embed_list]
        # Now reshape the embeddings so they have the shape B x M x E where B is the batch_size
        padded_embeddings = torch.stack(padded_embeddings).view(batch_size, max_embed_length, self.input_size)
        # keys is of shape L x B x F - reshape it so that we have B x F x L
        keys = keys.permute(1, 2, 0)
        # reshape values so that it has the shape B x Lx F
        values = values.permute(1, 0, 2)
        # creating tensor to keep track of predictions at each time step
        all_predictions = torch.zeros((batch_size, max_embed_length, self.vocab_size)).to(device)
        # creating attention parameter:
        # attention = self.attention.repeat((batch_size, 1, max_embed_length))
        # create attention matrix to visualize attention for some instance
        all_attention = torch.zeros((batch_size, max_embed_length, keys.size(2))).to(device)
        # now loop through the time steps M
        prediction = torch.zeros(batch_size, dtype=torch.long).to(device)
        # creating initial hidden state for lstmcell
        hidden = (h, h)
        h1, h2, h3 = hidden, hidden, hidden
        for t in range(max_embed_length):
            current_embedding = padded_embeddings[:, t, :].unsqueeze(1)
            # Implementing teacher forcing
            tf = np.random.uniform()
            if tf <= 0.1:
                current_embedding = self.embedding(prediction).unsqueeze(1)
                # current_embedding is now of size B x 1 x E
            # concat context with embedding along the last dimension, i.e., get F + E
            decoder_out = torch.cat((current_embedding, context), dim=2)
            # squeezing the singleton dimension in the middle so decoder_out has shape B x F+E currently
            decoder_out = decoder_out.squeeze(1)
            '''
            for i in range(self.nlayers):
                decoder_out, _ = self.rnncells[i](decoder_out, hidden)
            '''
            h1 = self.rnncells[0](decoder_out, h1)
            hh1, hc1 = h1
            h2 = self.rnncells[1](hh1, h2)
            hh2, hc2 = h2
            h3 = self.rnncells[2](hh2, h3)
            hh3, hc3 = h3
            # The output of the decoder is going to have shape B x H where H is the hidden size
            # Project this output onto a linear layer with output size F to get query B x F
            query = self.projection(hh3)
            # unsqueeze to get back the singleton dimension
            query = query.unsqueeze(1)
            # calculate query now to get B x 1 x L
            energy = torch.bmm(query, keys)
            # softmax along L to get probabilities
            attention = F.softmax(energy, dim=2)
            # plot attention to check if it's right
            all_attention[:, t, :] = attention[:, 0, :]
            # finally, calculate context using values - comes back to B x 1 x F
            context = torch.bmm(attention, values)
            # concatenating query with context before sending to scoring layer
            char_dist = torch.cat((query, context), dim=2)
            # predicting at each time step: this will be of size B x V where V is the vocab_size
            scoring_out = self.scoring(char_dist.view(-1, 2*self.key_size))
            # take the argmax across dimension V to get the prediction: we get B x 1
            _, prediction = scoring_out.max(1)
            # to send predictions through the loss function, we need the actual output values
            all_predictions[:, t, :] = scoring_out

        return all_predictions, all_attention

    def generate(self, start_label, keys, values, max_embed_length):
        # start_label is a single number -> so batch size of 1
        batch_size = 1
        # expand learnable hidden state parameter - now you have B x H
        h = self.h.repeat(batch_size, 1)
        # project using projection layer to get context as B x F
        context = self.projection(h)
        # unsqueeze context so you have B x 1 X F
        context = context.unsqueeze(1)
        # embedding will give us an embedding of size 1 x E
        embedded_label = self.embedding(start_label).unsqueeze(0)
        # keys is of shape L x B x F - reshape it so that we have B x F x L
        keys = keys.permute(1, 2, 0)
        # reshape values so that it has the shape B x Lx F
        values = values.permute(1, 0, 2)
        # # context is of dimension B x 1 x F - initialize context to be zeros
        # context = self.projection(h)
        # creating tensor to keep track of predictions at each time step
        all_predictions = torch.zeros(max_embed_length, dtype=torch.long).to(device)
        scoring = torch.zeros((max_embed_length, batch_size, self.vocab_size), dtype=torch.float).to(device)
        # creating attention parameter:
        # generate sequence until either you get to an EOS character or for 50 time steps
        t = 0
        prediction = torch.LongTensor([int(vocab['SOS'])])
        hidden = (h, h)
        h1, h2, h3 = hidden, hidden, hidden
        while (prediction != int(vocab['EOS'])) and (t < max_embed_length):
            if t != 0:
                embedded_label = self.embedding(prediction).unsqueeze(1)
            else:
                embedded_label = embedded_label.unsqueeze(1)

            decoder_out = torch.cat((embedded_label, context), dim=2)
            # squeezing the singleton dimension in the middle so decoder_out has shape B x F+E currently
            decoder_out = decoder_out.squeeze(1)
            '''
            for i in range(self.nlayers):
                decoder_out, _ = self.rnncells[i](decoder_out, hidden)
            '''
            h1 = self.rnncells[0](decoder_out, h1)
            hh1, hc1 = h1
            h2 = self.rnncells[1](hh1, h2)
            hh2, hc2 = h2
            h3 = self.rnncells[2](hh2, h3)
            hh3, hc3 = h3
            # The output of the decoder is going to have shape B x H where H is the hidden size
            # Project this output onto a linear layer with output size F to get query B x F
            query = self.projection(hh3)
            # unsqueeze to get back the singleton dimension
            query = query.unsqueeze(1)
            # calculate query now to get B x 1 x L
            energy = torch.bmm(query, keys)
            # softmax along L to get probabilities
            attention = F.softmax(energy, dim=2)
            # finally, calculate context using values - comes back to B x 1 x F
            context = torch.bmm(attention, values)
            # concatenating query with context before sending to scoring layer
            char_dist = torch.cat((query, context), dim=2)
            # predicting at each time step: this will be of size B x V where V is the vocab_size
            scoring_out = self.scoring(char_dist.view(-1, 2*self.key_size))
            # adding gumbel noise
            # gumbel = torch.tensor(np.log(np.random.uniform(0, 1, (batch_size, self.vocab_size)))).float().to(device)
            # scoring_out = scoring_out + gumbel
            # take the argmax across dimension V to get the prediction: we get B x 1
            _, prediction = scoring_out.max(1)
            # these will act as ground truths so it's okay to send the argmax
            all_predictions[t] = prediction.view(-1)
            scoring[t, :, :] = scoring_out
            t += 1

        return all_predictions, t



def train(listener, speller, criterion, optimizer, train_loader, epoch):
    start = time.time()
    print("Training: ")
    outfile.write("Training: \n")
    avg_loss = 0
    avg_ppl = 0
    avg_dist = 0
    listener.train()
    speller.train()
    for i, (data, input, labels) in enumerate(train_loader):
        data_time = time.time() - start
        key, value = listener(data)
        output, attention = speller(input, key, value)

        if i==0:
            if (epoch+1)%2==0:
                plot_attention(attention)
        _, predictions = output.max(dim=2)
        label_lengths = [len(label) for label in labels]
        max_length = max(label_lengths)
        edit_dist = get_dist(predictions, labels, label_lengths)
        avg_dist += edit_dist
        padded_labels = torch.cat([F.pad(label, (0, max_length - len(label)), value=-100) for label in labels])
        loss = criterion(output.view(-1, output.size(2)), padded_labels.view(-1))
        loss = loss/len(data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        perplexity = np.exp(loss.item()/max_length)
        avg_ppl += perplexity
        batch_time = time.time() - start
        if (i+1)%args.print_freq==0:
            print('Epoch: [{}][{}/{}]\t'
              'Loss: {:.3f}\t'
              'Perplexity: {:.3f}\t'
              'Distance: {:.3f}\t'
              'Data Time: {:.3f}\t'
              'Batch Time: {:.3f}'.format(epoch, i, len(train_loader),\
               avg_loss/(i+1), avg_ppl/(i+1), avg_dist/(i+1), data_time, batch_time))
            outfile.write('Epoch: [{}][{}/{}]\t'
              'Loss: {:.3f}\t'
              'Perplexity: {:.3f}\t'
              'Distance: {:.3f}\t'
              'Data Time: {:.3f}\t'
              'Batch Time: {:.3f}'.format(epoch, i, len(train_loader),\
               avg_loss/(i+1), avg_ppl/(i+1), avg_dist/(i+1), data_time, batch_time))
            outfile.write('\n')

    return avg_loss/len(train_loader), avg_ppl/len(train_loader)

def validate(listener, speller, criterion, dev_loader, epoch):
    start = time.time()
    print("Validation: ")
    outfile.write("Validation: \n")
    avg_loss = 0
    avg_ppl = 0
    avg_dist = 0
    listener.eval()
    speller.eval()
    with torch.no_grad():
        for i, (data, input, labels) in enumerate(dev_loader):
            data_time = time.time() - start
            key, value = listener(data)
            output, attention = speller(input, key, value)
            if i == 0:
                if (epoch+1)%2==0:
                    plot_attention(attention)
            _, predictions = output.max(dim=2)
            label_lengths = [len(label) for label in labels]
            max_length = max(label_lengths)
            edit_dist = get_dist(predictions, labels, label_lengths)
            avg_dist += edit_dist
            padded_labels = torch.cat([F.pad(label, (0, max_length - len(label)), value=-100) for label in labels])
            loss = criterion(output.view(-1, output.size(2)), padded_labels.view(-1))
            loss = loss/len(data)
            avg_loss += loss.item()
            perplexity = np.exp(loss.item()/max_length)
            avg_ppl += perplexity
            batch_time = time.time() - start
            if (i+1)%(args.print_freq//3)==0:
                print('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'Perplexity: {:.3f}\t'
                  'Distance: {:.3f}\t'
                  'Data Time: {:.3f}\t'
                  'Batch Time: {:.3f}'.format(epoch, i, len(dev_loader),\
                   avg_loss/(i+1), avg_ppl/(i+1), avg_dist/(i+1), data_time, batch_time))
                outfile.write('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.3f}\t'
                  'Perplexity: {:.3f}\t'
                  'Distance: {:.3f}\t'
                  'Data Time: {:.3f}\t'
                  'Batch Time: {:.3f}'.format(epoch, i, len(dev_loader),\
                   avg_loss/(i+1), avg_ppl/(i+1), avg_dist/(i+1), data_time, batch_time))
                outfile.write('\n')


    return avg_loss/len(dev_loader), avg_ppl/len(dev_loader)
'''

def test(listener, speller, criterion, test_loader):
    print("Testing: ")
    outfile.write("Testing:\n")
    test_predictions = []
    listener.eval()
    speller.eval()
    start_label = torch.tensor(int(vocab['SOS']), dtype=torch.long).to(device)
    test_predictions = []
    gt_batch_size = 500
    max_seq_length = 250
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    final_sequence = None
    prev_j = math.inf
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print('[{}]/[{}]'.format(i+1, len(test_loader)))
            outfile.write('[{}]/[{}]'.format(i+1, len(test_loader)))
            outfile.write('\n')
            keys, values = listener(data)
            bt_keys = keys.repeat(1, gt_batch_size, 1)
            bt_values = values.repeat(1, gt_batch_size, 1)
            labels = []
            ground_truths = []
            best_loss = math.inf
            for j in range(gt_batch_size):
                # generating 500 sequences for a single test utterance
                # ground_truth is of size L since batch size is just 1
                ground_truth, seq_length = speller.generate(start_label, keys, values, max_seq_length)
                # maintaing the original sequence length of the generated sequence and ensuring it only goes up to :-1
                ground_truths.append(ground_truth[:seq_length-1])
                # also need "labels" to compare generated sequence with
                labels.append(ground_truth[1:seq_length])
                # generated_sequence will be of size B x L x V or 1 x L x V

            generated_sequence, attention = speller(ground_truths, bt_keys, bt_values, isTest=True)

            plot_attention(attention, name='test_attention.png')
            # No need to pad generated sequence since it is just a single example
            # # pad ground_truths so all "labels" have the same length
            max_label_length = max([len(label) for label in labels])
            padded_labels = torch.cat([F.pad(label, (0, max_label_length - len(label)), value=-100) for label in labels])
            loss = criterion(generated_sequence.view(-1, generated_sequence.size(2)), padded_labels.view(-1))
            loss = loss.view(gt_batch_size, -1)
            loss = torch.sum(loss, dim=1)
            min_loss, index = torch.min(loss, 0)
            # finding the ground truth that got the minimum loss
            test_ppl = np.exp(min_loss.item())
            final_sequence = ground_truths[int(index)]
            sentence = ''
            for s in range(len(final_sequence)):
                if devocab[int(final_sequence[s].item())] != 'SOS' and devocab[int(final_sequence[s].item())] != 'EOS':
                    sentence += devocab[int(final_sequence[s])]

            print("Generated sentence: ", sentence)

            test_predictions.append(sentence)
    np.save(os.path.join(args.foldername, 'predictions.npy'), np.asarray(test_predictions))



    write_to_csv(test_predictions)
'''
def test(listener, speller, criterion, test_loader):
    print("Testing: ")
    outfile.write("Testing:\n")
    test_predictions = []
    listener.eval()
    speller.eval()
    start_label = torch.tensor(int(vocab['SOS']), dtype=torch.long).to(device)
    test_predictions = []
    max_seq_length = 250
    final_sequence = None
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print('[{}]/[{}]'.format(i+1, len(test_loader)))
            outfile.write('[{}]/[{}]'.format(i+1, len(test_loader)))
            outfile.write('\n')
            keys, values = listener(data)

            ground_truth, seq_length = speller.generate(start_label, keys, values, max_seq_length)
            sentence = ''
            for s in range(len(ground_truth)):
                if devocab[int(ground_truth[s].item())] != 'SOS' and devocab[int(ground_truth[s].item())] != 'EOS':
                    sentence += devocab[int(ground_truth[s])]
            print("Generated sentence: ", sentence)

            test_predictions.append(sentence)
    np.save(os.path.join(args.foldername, 'predictions.npy'), np.asarray(test_predictions))
    write_to_csv(test_predictions)
def main():
    '''
    ####################### TESTING ON TOY DATA ##################################################################
    hidden_size = 5
    input_size = 3
    out_features = 4
    a = torch.tensor(np.random.randint(0, 10, (30, 3))).float()
    b = torch.tensor(np.random.randint(0, 10, (20, 3))).float()
    c = torch.tensor(np.random.randint(0, 10, (12, 3))).float()
    label_a = torch.tensor(np.random.randint(0, 10, 15)).long()
    label_b = torch.tensor(np.random.randint(0, 10, 12)).long()
    label_c = torch.tensor(np.random.randint(0, 10, 7)).long()
    utterance_list = [a, b, c]
    labels = [label_a, label_b, label_c]
    listener = Encoder(hidden_size, input_size, out_features)
    speller = Decoder(input_size, hidden_size, 1, out_features, 10)
    key, value = listener(utterance_list)
    speller.eval()
    print(key.shape, value.shape)
    output, lengths = speller.generate(labels[0][0], key[:, 0, :].unsqueeze(1), value[:, 0, :].unsqueeze(1), 5)
    ###################################################################################################################
    '''
    encoder_hidden_size = 128
    encoder_input_size = 40
    encoder_out_features = 64

    decoder_hidden_size = 256
    decoder_input_size = 128
    decoder_key_size = 64

    vocab_size = len(vocab.keys())
    # defining listener and speller
    listener = Encoder(hidden_size=encoder_hidden_size, input_size=encoder_input_size, nlayers=4, out_features=encoder_out_features)
    speller = Decoder(hidden_size=decoder_hidden_size, input_size=decoder_input_size, key_size=decoder_key_size, nlayers=3, vocab_size=vocab_size)
    listener.to(device)
    speller.to(device)
    if args.pretrain:
        spl_checkpoint = torch.load(os.path.join(args.pretrain, 'pretrained_speller.pth.tar'))
        speller.load_state_dict(spl_checkpoint['state_dict'])
    # loading data
    train_data = np.load(os.path.join(args.data_path, 'train.npy'), encoding='bytes')
    train_transcripts = np.load(os.path.join(args.data_path, 'train_transcripts_labels.npy'))
    dev_data = np.load(os.path.join(args.data_path, 'dev.npy'), encoding='bytes')
    dev_transcripts = np.load(os.path.join(args.data_path, 'dev_transcripts_labels.npy'))
    test_data = np.load(os.path.join(args.data_path, 'test.npy'), encoding='bytes')

    # loading dataset
    train_dataset = framesDataset(frames=train_data, labels=train_transcripts)
    dev_dataset = framesDataset(frames=dev_data, labels=dev_transcripts)
    test_dataset = framesDataset(frames=test_data, labels=None)

    # using DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_frames)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_frames)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # defining criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum').to(device)
    optimizer = torch.optim.Adam(list(listener.parameters())+list(speller.parameters()), lr=args.lr)
    val_loss_all = []
    val_ppl_all = []
    train_loss_all = []
    train_ppl_all = []
    best_ppl = math.inf
    if not args.eval:
        if args.finetune:
            lst_checkpoint = torch.load(os.path.join(args.load_folder,'listener.pth.tar'))
            spl_checkpoint = torch.load(os.path.join(args.load_folder, 'speller.pth.tar'))
            listener.load_state_dict(lst_checkpoint['state_dict'])
            speller.load_state_dict(spl_checkpoint['state_dict'])
        for epoch in range(args.epochs):
            adjust_learning_rate(optimizer, epoch)
            train_loss, train_ppl = train(listener, speller, criterion, optimizer, train_loader, epoch)
            val_loss, val_ppl = validate(listener, speller, criterion, dev_loader, epoch)
            train_loss_all.append(train_loss)
            train_ppl_all.append(train_ppl)
            val_loss_all.append(val_loss)
            val_ppl_all.append(val_ppl)
            if best_ppl > val_ppl:
                best_ppl = val_ppl
                lst_checkpoint = {'state_dict': listener.state_dict()}
                spl_checkpoint = {'state_dict': speller.state_dict()}
                torch.save(lst_checkpoint, os.path.join(args.foldername, 'listener.pth.tar'))
                torch.save(spl_checkpoint, os.path.join(args.foldername, 'speller.pth.tar'))

        x = np.arange(args.epochs).tolist()
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].set_title('Train and Validation Loss')
        ax[0].plot(x, train_loss_all, label='Train Loss')
        ax[0].plot(x, val_loss_all, label='Val Loss' )
        ax[0].legend(loc='best')

        ax[1].set_title('Train and Validation Perplexity')
        ax[1].plot(x, train_ppl_all, label='Train PPL')
        ax[1].plot(x, val_ppl_all, label='Val PPL' )
        ax[1].legend(loc='best')

        plt.savefig(os.path.join(args.foldername, 'loss_and_ppl.png'))

    lst_checkpoint = torch.load(os.path.join(args.foldername,'listener.pth.tar'))
    spl_checkpoint = torch.load(os.path.join(args.foldername, 'speller.pth.tar'))
    listener.load_state_dict(lst_checkpoint['state_dict'])
    speller.load_state_dict(spl_checkpoint['state_dict'])
    '''
    for name, param in listener.named_parameters():
        if 'weight' in name:
            print('Encoder')
            s = torch.sum(param.data, dim=1)
            s = torch.sum(s, dim=0)
            max_val, _ = torch.max(param.data, dim=1)
            max_val, _ = torch.max(max_val, dim=0)
            min_val, _ = torch.min(param.data, dim=1)
            min_val, _ = torch.min(min_val, dim=0)
            print(name, s, min_val, max_val)

    for name, param in speller.named_parameters():
        if 'weight' in name:
            print('Decoder')
            s = torch.sum(param.data, dim=1)
            s = torch.sum(s, dim=0)
            max_val, _ = torch.max(param.data, dim=1)
            max_val, _ = torch.max(max_val, dim=0)
            min_val, _ = torch.min(param.data, dim=1)
            min_val, _ = torch.min(min_val, dim=0)
            print(name, s, min_val, max_val)
    '''
    test(listener, speller, criterion, test_loader)

if __name__=='__main__':
    main()
