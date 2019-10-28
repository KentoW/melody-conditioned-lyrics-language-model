# -*- coding: utf-8 -*-
import sys
import json
import time
import datetime
import argparse
from data import SongDataset, collate_fn
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data as data
from torch.nn.utils.rnn import pack_padded_sequence
from collections import defaultdict
from model import MCLLM


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class LogPrint:
    def __init__(self, file_path, err):
        self.file = open(file_path, "w", buffering=1)
        self.err = err

    def lprint(self, text, ret=False, ret2=False):
        if self.err:
            if ret == True:
                if ret2 == True:
                    sys.stderr.write("\n" + text + "\n")
                else:
                    sys.stderr.write("\r" + text + "\n")
            else:
                sys.stderr.write("\r" + text)
        self.file.write(text + "\n")

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def main(args):
    """ Set the random seed manually for reproducibility """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Load Data """
    data_set = SongDataset(data=args.data, word_size=args.word_size, window=args.window)
    word_size = data_set.word_size
    feature_size = data_set.feature_size
    syllable_size = data_set.syllable_size
    lp.lprint("------ Data Stats -----", True)
    lp.lprint("{:>12}:  {}".format("song size", len(data_set)), True)
    lp.lprint("{:>12}:  {}".format("vocab size", word_size), True)
    lp.lprint("{:>12}:  {}".format("feature size", feature_size), True)
    lp.lprint("{:>12}:  {}".format("syllable size", syllable_size), True)

    """ write vocab and model param"""
    with open(args.checkpoint+".feature.json", 'w') as f:
        f.write(json.dumps(data_set.idx2feature, ensure_ascii=False))

    with open(args.checkpoint+".vocab.json", 'w') as f:
        f.write(json.dumps(data_set.idx2word, ensure_ascii=False))

    with open(args.checkpoint+".param.json", 'w') as f:
        f.write(json.dumps({"feature_idx_path":args.checkpoint+".feature.json", 
                            "vocab_idx_path":args.checkpoint+".vocab.json", 
                            "word_dim":args.word_dim, 
                            "melody_dim":args.melody_dim, 
                            "syllable_size":syllable_size, 
                            "feature_size":feature_size, 
                            "window":args.window, 
                            "args_word_size":args.word_size}, ensure_ascii=False))

    """ split train/valid """
    n_samples = len(data_set)
    train_size = int(len(data_set) * 0.9)
    val_size = n_samples - train_size
    train_data_set, val_data_set = torch.utils.data.random_split(data_set, [train_size, val_size])

    """ make data loader """
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data_set, 
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers, 
                                              collate_fn=collate_fn)

    val_data_loader = torch.utils.data.DataLoader(dataset=val_data_set, 
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers, 
                                              collate_fn=collate_fn)

    """ Build the model """
    model = MCLLM(word_dim=args.word_dim, melody_dim=args.melody_dim, syllable_size=syllable_size, word_size=word_size, feature_size=feature_size).to(device)

    """ Optimizer """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    lr = args.lr
    criterion = nn.CrossEntropyLoss()

    """ Define training function """
    def train(epoch, data_set, data_loader):
        model.train()
        """ Logging time """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        sum_losses_s = AverageMeter()
        sum_losses_l = AverageMeter()
        start = time.time()
        """ Batches """
        hidden = model.init_hidden(args.batch_size)
        for i, (syllable, lyrics, melody, lengths) in enumerate(data_loader):
            data_time.update(time.time()*1000 - start*1000)
            """ Move to GPU """
            syllable = syllable.to(device)
            lyrics = lyrics.to(device)
            melody = melody.to(device).float()
            lengths = lengths.to(device)
            """ Remove first melody feature """
            melody = melody[:, 1:]

            """ Zero Grad """
            optimizer.zero_grad()
            """ detach """
            hidden = repackage_hidden(hidden)

            """ forward """
            syllable_output, lyrics_output, hidden = model(lyrics[:, :-1], melody, lengths)
            target_syllable = pack_padded_sequence(syllable[:, 1:], lengths-1, batch_first=True)[0]
            target_lyrics = pack_padded_sequence(lyrics[:, 1:], lengths-1, batch_first=True)[0]
            loss_syllable = criterion(syllable_output, target_syllable)
            sum_losses_s.update(loss_syllable)
            loss_lyrics = criterion(lyrics_output, target_lyrics)
            sum_losses_l.update(loss_lyrics)
            """ Propagation """
            loss = loss_syllable + loss_lyrics
            loss.backward()
            optimizer.step()

            """ Keep track of metrics """
            batch_time.update(time.time()*1000 - start*1000)
            start = time.time()
            """ Print status """
            if i % args.log_interval == 0:
                lp.lprint('| Training Epoch: {:3d}/{:3d}  {:6d}/{:6d} '
                          '| lr:{:6.5f} '
                          '| {batch_time.avg:7.2f} ms/batch '
                          '| {data_time.avg:5.2f} ms/data_load '
                          '| Loss(Syllable) {loss_s.avg:5.5f} '
                          '| Loss(Lyrics) {loss_l.avg:5.5f} |'
                          .format(epoch+1, args.num_epochs, i, len(data_loader), lr, 
                                  batch_time=batch_time,
                                  data_time=data_time, 
                                  loss_s=sum_losses_s, 
                                  loss_l=sum_losses_l))

    """ Define validation function """
    def validation(epoch, data_set, data_loader):
        model.eval()
        """ Logging time """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        sum_losses_s = AverageMeter()
        sum_losses_l = AverageMeter()
        start = time.time()
        """ Batches """
        hidden = model.init_hidden(args.batch_size)
        for i, (syllable, lyrics, melody, lengths) in enumerate(data_loader):
            data_time.update(time.time()*1000 - start*1000)
            """ Move to GPU """
            syllable = syllable.to(device)
            lyrics = lyrics.to(device)
            melody = melody.to(device).float()
            lengths = lengths.to(device)
            """ Remove first melody feature """
            melody = melody[:, 1:]

            """ detach """
            hidden = repackage_hidden(hidden)
            """ forward """
            syllable_output, lyrics_output, hidden = model(lyrics[:, :-1], melody, lengths)
            target_syllable = pack_padded_sequence(syllable[:, 1:], lengths-1, batch_first=True)[0]
            target_lyrics = pack_padded_sequence(lyrics[:, 1:], lengths-1, batch_first=True)[0]
            loss_syllable = criterion(syllable_output, target_syllable)
            sum_losses_s.update(loss_syllable)
            loss_lyrics = criterion(lyrics_output, target_lyrics)
            sum_losses_l.update(loss_lyrics)

            """ Keep track of metrics """
            batch_time.update(time.time()*1000 - start*1000)
            start = time.time()
            """ Print status """
            if i % args.log_interval == 0:
                lp.lprint('| Validation Epoch: {:3d}/{:3d}  {:6d}/{:6d} '
                          '| lr:{:6.5f} '
                          '| {batch_time.avg:7.2f} ms/batch '
                          '| {data_time.avg:5.2f} ms/data_load '
                          '| Loss(Syllable) {loss_s.avg:5.5f} '
                          '| Loss(Lyrics) {loss_l.avg:5.5f} |'
                          .format(epoch+1, args.num_epochs, i, len(data_loader), lr, 
                                  batch_time=batch_time,
                                  data_time=data_time, 
                                  loss_s=sum_losses_s, 
                                  loss_l=sum_losses_l))

    def save_model(epoch):
        model.eval()
        with open(args.checkpoint+"_%02d.pt"%(epoch+1), 'wb') as f:
            torch.save(model.state_dict(), f)

    """ Epochs """
    lp.lprint("------ Training -----", True)
    for epoch in range(args.num_epochs):
        """ Training """
        train(epoch, train_data_set, train_data_loader)
        lp.lprint("", True)
        """ Validation and save checkpoint """
        with torch.no_grad():
            validation(epoch, val_data_set, val_data_loader)
            lp.lprint("", True)
            """ Save checkpoint """
            save_model(epoch)
        lp.lprint("-----------", True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data parameter """
    parser.add_argument("-data", "--data", dest="data", default="./sample_data/pseudo_data.jsonl", type=str, help="alignment data")
    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./checkpoint/model", type=str, help="save path")

    """ Feature parameter """
    parser.add_argument("-window", "--window", dest="window", default=10, type=int, help="window size of melody featrure (default: 10)")
    parser.add_argument("-word_size", "--word_size", dest="word_size", default=20000, type=int, help="vocab size (default: 20000)")

    """ Model parameter """
    parser.add_argument("-word_dim", "--word_dim", dest="word_dim", default=512, type=int, help="dimension of Word Embedding (default: 512)")
    parser.add_argument("-melody_dim", "--melody_dim", dest="melody_dim", default=256, type=int, help="dimension of Melody Layer (default: 256)")

    """ Training parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="random seed")
    parser.add_argument("-num_workers", "--num_workers", dest="num_workers", default=4, type=int, help="number of CPU")
    parser.add_argument("-num_epochs", "--num_epochs", dest="num_epochs", default=5, type=int, help="Epochs")
    parser.add_argument("-batch_size", "--batch_size", dest="batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("-lr", "--lr", dest="lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("-log_interval", "--log_interval", dest="log_interval", default=10, type=int, help="Report interval")

    """ Logging parameter """
    parser.add_argument("-verbose", "--verbose", dest="verbose", default=1, type=int, help="verbose 0/1")
    args = parser.parse_args()

    """ Save parameter """
    if args.verbose == 1:
        lp = LogPrint(args.checkpoint + ".log", True)
    else:
        lp = LogPrint(args.checkpoint + ".log", False)
    argparse_dict = vars(args)
    lp.lprint("------ Parameters -----", True)
    for k, v in argparse_dict.items():
        lp.lprint("{:>16}:  {}".format(k, v), True)
    with open(args.checkpoint+".args.json", 'w') as f:
        f.write(json.dumps(argparse_dict))
    main(args)





