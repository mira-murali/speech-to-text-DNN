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
parser.add_argument('--print-freq', default=20, type=int, help='how often performance must be printed per epoch')

args = parser.parse_args()
dir_path = os.path.realpath('pretrain_speller.py')
os.environ['CURRENT'] = dir_path[:dir_path.find('pretrain')]
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
    for i in range(len(predictions)):
        sentence = predictions[i]
        pred_writer.writerow([str(i), sentence])

class labelsDataset(Dataset):
    def __init__(self, labels):
        self.labels = [torch.LongTensor(labels[i]) for i in range(labels.shape[0])]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input = self.labels[idx][:-1]
        target = self.labels[idx][1:]
        return input.to(device), target.to(device)

def collate_frames(utterance_list):
    inputs, targets = zip(*utterance_list)
    seq_lengths = [len(inputs) for input in inputs]
    seq_order = sorted(range(len(seq_lengths)), key=seq_lengths.__getitem__, reverse=True)
    sorted_targets = [targets[i] for i in seq_order]
    sorted_inputs = [inputs[i] for i in seq_order]
    return sorted_inputs, sorted_targets

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
         self.scoring = nn.Linear(2*self.key_size, self.vocab_size)
         # making attention a learnable parameter
         self.attention = nn.Parameter(torch.zeros(1, 1, 1), requires_grad = True).to(device)
         # making hidden state a learnable parameter
         self.h = nn.Parameter(torch.zeros((1, self.hidden_size)), requires_grad=True).to(device)

    def weight_init(self, layer):
        if layer == nn.Embedding:
            nn.init.uniform_(layer.weight, -0.1, 0.1)
        if layer == nn.Linear:
            nn.init.uniform_(layer.weight, -np.sqrt(1/self.hidden_size), np.sqrt(1/self.hidden_size))

    def forward(self, labels, isTest=False):
        batch_size = len(labels)
        # expand learnable hidden state parameter - now you have B x H
        h = self.h.repeat(batch_size, 1)
        # force context to be zeros when pretraining
        context = self.projection(h).unsqueeze(1)
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
        # creating tensor to keep track of predictions at each time step
        all_predictions = torch.zeros((batch_size, max_embed_length, self.vocab_size)).to(device)
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

            h1 = self.rnncells[0](decoder_out, h1)
            hh1, hc1 = h1
            h2 = self.rnncells[1](hh1, h2)
            hh2, hc2 = h2
            h3 = self.rnncells[2](hh2, h3)
            hh3, hc3 = h3
            # The output of the decoder is going to have shape B x H where H is the hidden size
            # Project this output onto a linear layer with output size F to get query B x F
            query = self.projection(hh3)
            char_dist = torch.cat((query, context.squeeze(1)), dim=1)
            # predicting at each time step: this will be of size B x V where V is the vocab_size
            scoring_out = self.scoring(char_dist.view(-1, 2*self.key_size))
            # take the argmax across dimension V to get the prediction: we get B x 1
            _, prediction = scoring_out.max(1)
            # to send predictions through the loss function, we need the actual output values
            all_predictions[:, t, :] = scoring_out

        return all_predictions

    def generate(self, start_label, max_embed_length):
        # start_label is a single number -> so batch size of 1
        batch_size = 1
        # expand learnable hidden state parameter - now you have B x H
        h = self.h.repeat(batch_size, 1)
        # context must always be zeros
        context = torch.zeros((batch_size, 1, self.key_size)).to(device)
        # embedding will give us an embedding of size 1 x E
        embedded_label = self.embedding(start_label).unsqueeze(0)
        # creating tensor to keep track of predictions at each time step
        all_predictions = torch.zeros(max_embed_length, dtype=torch.long).to(device)
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
            h1 = self.rnncells[0](decoder_out, h1)
            hh1, hc1 = h1
            h2 = self.rnncells[1](hh1, h2)
            hh2, hc2 = h2
            h3 = self.rnncells[2](hh2, h3)
            hh3, hc3 = h3
            # The output of the decoder is going to have shape B x H where H is the hidden size
            # Project this output onto a linear layer with output size F to get query B x F
            query = self.projection(hh3)
            # concatenate query with context
            char_dist = torch.cat((query, context.squeeze(1)), dim=1)
            # predicting at each time step: this will be of size B x V where V is the vocab_size
            scoring_out = self.scoring(char_dist.view(-1, 2*self.key_size))
            # take the argmax across dimension V to get the prediction: we get B x 1
            _, prediction = scoring_out.max(1)
            # these will act as ground truths so it's okay to send the argmax
            all_predictions[t] = prediction.view(-1)
            t += 1

        return all_predictions, t



def train(speller, criterion, optimizer, train_loader, epoch):
    start = time.time()
    print("Training: ")
    outfile.write("Training: \n")
    avg_loss = 0
    avg_ppl = 0
    avg_dist = 0
    speller.train()
    for i, (input, labels) in enumerate(train_loader):
        data_time = time.time() - start
        output = speller(input)
        _, predictions = output.max(dim=2)
        label_lengths = [len(label) for label in labels]
        max_length = max(label_lengths)
        edit_dist = get_dist(predictions, labels, label_lengths)
        avg_dist += edit_dist
        padded_labels = torch.cat([F.pad(label, (0, max_length - len(label)), value=-100) for label in labels])
        loss = criterion(output.view(-1, output.size(2)), padded_labels.view(-1))
        # loss will be a tensor of size N*L
        loss = loss/len(input)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(speller.parameters(), 5.0)
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

def validate(speller, criterion, dev_loader, epoch):
    start = time.time()
    print("Validation: ")
    outfile.write("Validation: \n")
    avg_loss = 0
    avg_ppl = 0
    avg_dist = 0
    speller.eval()
    with torch.no_grad():
        for i, (input, labels) in enumerate(dev_loader):
            data_time = time.time() - start
            output = speller(input)
            _, predictions = output.max(dim=2)
            label_lengths = [len(label) for label in labels]
            max_length = max(label_lengths)
            edit_dist = get_dist(predictions, labels, label_lengths)
            avg_dist += edit_dist
            padded_labels = torch.cat([F.pad(label, (0, max_length - len(label)), value=-100) for label in labels])
            loss = criterion(output.view(-1, output.size(2)), padded_labels.view(-1))
            loss = loss/len(input)
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


def test(speller, test_loader):
    print("Testing: ")
    outfile.write("Testing:\n")
    test_predictions = []
    speller.eval()
    start_label = torch.tensor(int(vocab['SOS']), dtype=torch.long).to(device)
    test_predictions = []
    gt_batch_size = 50
    max_seq_length = 250
    best_loss = math.inf
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    final_sequence = None
    with torch.no_grad():
        for i, (input, targets) in enumerate(test_loader):
            print('[{}]/[{}]'.format(i+1, len(test_loader)))
            outfile.write('[{}]/[{}]'.format(i+1, len(test_loader)))
            outfile.write('\n')
            labels = []
            ground_truths = []
            best_loss = math.inf
            for j in range(gt_batch_size):
                # generating 500 sequences for a single test utterance
                # ground_truth is of size L since batch size is just 1
                ground_truth, seq_length = speller.generate(start_label, max_seq_length)
                # maintaing the original sequence length of the generated sequence and ensuring it only goes up to :-1
                ground_truths.append(ground_truth[:seq_length-1])
                # also need "labels" to compare generated sequence with
                labels.append(ground_truth[1:seq_length])

            # generated_sequence will be of size B x L x V
            generated_sequence = speller(ground_truths, isTest=True)
            # pad ground_truths so all "labels" have the same length
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
    speller = Decoder(hidden_size=decoder_hidden_size, input_size=decoder_input_size, key_size=decoder_key_size, nlayers=3, vocab_size=vocab_size)
    speller.apply(speller.weight_init)
    speller.to(device)

    train_transcripts = np.load(os.path.join(args.data_path, 'train_transcripts_labels.npy'))
    dev_transcripts = np.load(os.path.join(args.data_path, 'dev_transcripts_labels.npy'))

    # loading dataset
    train_dataset = labelsDataset(labels=train_transcripts)
    dev_dataset = labelsDataset(labels=dev_transcripts)
    test_dataset = labelsDataset(labels=dev_transcripts)

    # using DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_frames)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_frames)
    test_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    # defining criterion and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum').to(device)
    optimizer = torch.optim.Adam(speller.parameters(), lr=args.lr)

    val_loss_all = []
    val_ppl_all = []
    train_loss_all = []
    train_ppl_all = []
    best_ppl = math.inf
    if not args.eval:
        if args.finetune:
            spl_checkpoint = torch.load(os.path.join(args.load_folder, 'pretrained_speller.pth.tar'))
            speller.load_state_dict(spl_checkpoint['state_dict'])
        for epoch in range(args.epochs):
            train_loss, train_ppl = train(speller, criterion, optimizer, train_loader, epoch)
            val_loss, val_ppl = validate(speller, criterion, dev_loader, epoch)
            train_loss_all.append(train_loss)
            train_ppl_all.append(train_ppl)
            val_loss_all.append(val_loss)
            val_ppl_all.append(val_ppl)
            if best_ppl > val_ppl:
                best_ppl = val_ppl
                spl_checkpoint = {'state_dict': speller.state_dict()}
                torch.save(spl_checkpoint, os.path.join(args.foldername, 'pretrained_speller.pth.tar'))

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

    spl_checkpoint = torch.load(os.path.join(args.foldername, 'pretrained_speller.pth.tar'))
    speller.load_state_dict(spl_checkpoint['state_dict'])
    test(speller, test_loader)

if __name__=='__main__':
    main()
