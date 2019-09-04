# -*- coding: utf-8 -*-
import sys
import random
import numpy as np
import json
import torch
import torch.utils.data as data
import glob
from collections import defaultdict

import logging
logging.disable(logging.FATAL)

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import re
en_p = re.compile(r"[a-zA-Z\']")

features = []
note_type=['rest', 'note']
lengths = [60, 120, 240, 360, 480, 720, 960, 1200, 1440, 1680, 1920, 3840, 5760, 7680]
tags = ['<WORD>', '<BOL>', '<BOB>']


def get_length(length):
    length = int(length)
    if length <= 80:
        return 60
    elif length <= 180:
        return 120
    elif length <= 300:
        return 240
    elif length <= 420:
        return 360
    elif length <= 600:
        return 480
    elif length <= 840:
        return 720
    elif length <= 1080:
        return 960
    elif length <= 1320:
        return 1200
    elif length <= 1560:
        return 1440
    elif length <= 1800:
        return 1680
    elif length <= 2880:
        return 1920
    elif length <= 4800:
        return 3840
    elif length <= 4560:
        return 5760
    else:
        return 7680

class SongDataset(data.Dataset):
    def __init__(self, data, word_size, window):
        """ make feature vocab """
        for i in range(window):
            for note_t in note_type:
                features.append('note[%s]=%s'%(i, note_t))
                features.append('note[%s]=%s'%(-(i+1), note_t))
            for length in lengths:
                features.append('length[%s]=%s'%(i, length))
                features.append('length[%s]=%s'%(-(i+1), length))
        for tag in tags:
            features.append('prev_tag=%s'%tag)
        Fs = sorted(features)
        self.feature2idx = dict((f, i) for i, f in enumerate(Fs))
        self.idx2feature = dict((i, f) for i, f in enumerate(Fs))
        self.feature_size = len(self.feature2idx)

        """ make word vocab and syllable size"""
        D = defaultdict(int)
        for strm in open(data, "r"):
            d = json.loads(strm.strip())
            old_word_idx = "<None>"
            for note in d["lyrics"]:
                word_idx = note[5]
                if word_idx != old_word_idx:
                    if type(note[4]) == list:
                        sur = note[4][0]
                        pos = note[4][1]
                        dtl = note[4][2]
                        mora = "_".join(note[4][3])
                        if en_p.match(sur):                 # NOTE ignore English words
                            continue
                        word = "%s,%s,%s|%s"%(sur, pos, dtl, mora)
                        D[word] += 1
                old_word_idx = word_idx
        self.word2idx = {}
        self.word2idx["<pad>"] = 0
        self.word2idx["<unk>"] = 1
        self.word2idx["<BOB>|<null>"] = 2
        self.word2idx["<BOL>|<null>"] = 3
        self.idx2word = {}
        self.idx2word[0] = "<pad>"
        self.idx2word[1] = "<unk>"
        self.idx2word[2] = "<BOB>|<null>"
        self.idx2word[3] = "<BOL>|<null>"
        idx = 4
        syllables = set()
        for word, freq in sorted(D.items(), key=lambda x:x[1], reverse=True)[:word_size:]:
            syllables.add(len(word.split("|")[-1].split("_")))
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1
        self.word_size = len(self.word2idx)
        self.syllable_size = max(syllables) + 10

        """ make instance """
        self.idx2syllable = []
        self.idx2lyrics = []
        self.idx2melody = []
        for strm in open(data, "r"):
            d = json.loads(strm.strip())
            old_word_idx = "<None>"
            tag_stack = ["<WORD>"]
            syllables = []
            lyrics = []
            melody = []

            for i, note in enumerate(d["lyrics"]):
                word_idx = note[5]
                if word_idx != old_word_idx:
                    if type(note[4]) == list:
                        sur = note[4][0]
                        pos = note[4][1]
                        dtl = note[4][2]
                        typ = note[6]
                        mora = "_".join(note[4][3])
                        syl = len(note[4][3]) + 1
                        word = "%s,%s,%s|%s"%(sur, pos, dtl, mora)

                        prev_i = i - window + 1
                        if prev_i < 0:
                            prev_i = 0
                        next_i = i + window
                        if next_i > len(d["lyrics"]):
                            next_i = len(d["lyrics"]) 
                        prev_notes = d["lyrics"][prev_i:i]
                        next_notes = d["lyrics"][i:next_i]

                        if typ == "<BOB>":
                            #print(prev_notes[-1][1], next_notes[0][1], tag_stack[-1], "<BOB>")
                            feature = []
                            w_idx = self.word2idx.get("<BOB>|<null>", self.word2idx["<unk>"])
                            lyrics.append(w_idx)
                            syllables.append(1)
                            prev_tag = self.feature2idx["prev_tag=%s"%tag_stack[-1]]
                            feature.append(prev_tag)
                            for j, pn in enumerate(prev_notes):
                                if pn[1] == "rest":
                                    note_num = self.feature2idx["note[-%s]=rest"%(len(prev_notes)-j)]
                                else:
                                    note_num = self.feature2idx["note[-%s]=note"%(len(prev_notes)-j)]
                                note_duration = self.feature2idx["length[-%s]=%s"%(len(prev_notes)-j, get_length(pn[2]))]
                                feature.append(note_num)
                                feature.append(note_duration)
                            for j, nn in enumerate(next_notes):
                                if nn[1] == "rest":
                                    note_num = self.feature2idx["note[%s]=rest"%j]
                                else:
                                    note_num = self.feature2idx["note[%s]=note"%j]
                                      
                                note_duration = self.feature2idx["length[%s]=%s"%(j, get_length(nn[2]))]
                                feature.append(note_num)
                                feature.append(note_duration)
                            feature = [feature[0]] * (39 - len(feature)) + feature      # NOTE pad feature
                            melody.append(feature[::])
                            tag_stack.append("<BOB>")
                        elif typ == "<BOL>":
                            #print(prev_notes[-1][1], next_notes[0][1], tag_stack[-1], "<BOL>")
                            feature = []
                            w_idx = self.word2idx.get("<BOL>|<null>", self.word2idx["<unk>"])
                            lyrics.append(w_idx)
                            syllables.append(1)
                            prev_tag = self.feature2idx["prev_tag=%s"%tag_stack[-1]]
                            feature.append(prev_tag)
                            for j, pn in enumerate(prev_notes):
                                if pn[1] == "rest":
                                    note_num = self.feature2idx["note[-%s]=rest"%(len(prev_notes)-j)]
                                else:
                                    note_num = self.feature2idx["note[-%s]=note"%(len(prev_notes)-j)]
                                note_duration = self.feature2idx["length[-%s]=%s"%(len(prev_notes)-j, get_length(pn[2]))]
                                feature.append(note_num)
                                feature.append(note_duration)
                            for j, nn in enumerate(next_notes):
                                if nn[1] == "rest":
                                    note_num = self.feature2idx["note[%s]=rest"%j]
                                else:
                                    note_num = self.feature2idx["note[%s]=note"%j]
                                      
                                note_duration = self.feature2idx["length[%s]=%s"%(j, get_length(nn[2]))]
                                feature.append(note_num)
                                feature.append(note_duration)
                            feature = [feature[0]] * (39 - len(feature)) + feature      # NOTE pad feature
                            melody.append(feature[::])
                            tag_stack.append("<BOL>")

                        #print(prev_notes[-1][1], next_notes[0][1], tag_stack[-1], word)
                        feature = []
                        w_idx = self.word2idx.get(word, self.word2idx["<unk>"])
                        lyrics.append(w_idx)
                        syllables.append(syl)
                        prev_tag = self.feature2idx["prev_tag=%s"%tag_stack[-1]]
                        feature.append(prev_tag)
                        for j, pn in enumerate(prev_notes):
                            if pn[1] == "rest":
                                note_num = self.feature2idx["note[-%s]=rest"%(len(prev_notes)-j)]
                            else:
                                note_num = self.feature2idx["note[-%s]=note"%(len(prev_notes)-j)]
                            note_duration = self.feature2idx["length[-%s]=%s"%(len(prev_notes)-j, get_length(pn[2]))]
                            feature.append(note_num)
                            feature.append(note_duration)
                        for j, nn in enumerate(next_notes):
                            if nn[1] == "rest":
                                note_num = self.feature2idx["note[%s]=rest"%j]
                            else:
                                note_num = self.feature2idx["note[%s]=note"%j]
                                  
                            note_duration = self.feature2idx["length[%s]=%s"%(j, get_length(nn[2]))]
                            feature.append(note_num)
                            feature.append(note_duration)
                        feature = [feature[0]] * (39 - len(feature)) + feature      # NOTE pad feature
                        melody.append(feature[::])
                        tag_stack.append("<WORD>")
                old_word_idx = word_idx

            self.idx2syllable.append(syllables[::])
            self.idx2lyrics.append(lyrics[::])
            self.idx2melody.append(melody[::])

    def __getitem__(self, idx):
        syllable = torch.Tensor(self.idx2syllable[idx])
        lyrics = torch.Tensor(self.idx2lyrics[idx])
        melody = self.idx2melody[idx]
        return syllable, lyrics, melody, self.feature_size


    def __len__(self):
        return len(self.idx2lyrics)


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    _syllable, _lyrics, _melody, feature_size = zip(*data)
    lengths = [len(_lyric) for _lyric in _lyrics]
    max_length = lengths[0]
    lyrics = torch.zeros(len(_lyrics), max_length).long()
    syllable = torch.zeros(len(_syllable), max_length).long()
    melody = torch.zeros(len(_melody), max_length, feature_size[0]).long()
    for i, _lyric in enumerate(_lyrics):
        end = lengths[i]
        lyrics[i, :end] = _lyric[:end]
        syllable[i, :end] = _syllable[i][:end]
        melody[i, :end].scatter_(1, torch.Tensor(_melody[i]).long(), 1)
    lengths = torch.Tensor(lengths).long()
    return syllable, lyrics, melody, lengths







