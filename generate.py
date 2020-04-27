# -*- coding: utf-8 -*-
import sys
import json
import argparse
import random
import math
import numpy as np
import jaconv

from util.convert_midi4generation import convert

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from model import MCLLM

import logging
logging.disable(logging.FATAL)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(notes, param, checkpoint, seed=0, window=2, temperature=1.0):
    """ Set the random seed manually for reproducibility """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Load parameters """
    params = json.loads(open(param, "r").readline())
    melody_dim = params["melody_dim"]
    word_dim = params["word_dim"]
    feature_window = params["window"]

    syllable_size = params["syllable_size"]

    idx2feature = json.loads(open(params["feature_idx_path"], "r").readline())
    feature2idx = dict([(v, int(k)) for k, v in idx2feature.items()])
    feature_size = len(feature2idx)

    idx2word = json.loads(open(params["vocab_idx_path"], "r").readline())
    word2idx = dict([(v, int(k)) for k, v in idx2word.items()])
    word_size = len(word2idx)
    bob = word2idx["<BOB>|<null>"]
    bol = word2idx["<BOL>|<null>"]

    """ Load model """
    model = MCLLM(word_dim=word_dim, melody_dim=melody_dim, syllable_size=syllable_size, word_size=word_size, feature_size=feature_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    hidden = model.init_hidden(1)

    """ define feature function """
    def make_feature(prev_notes, next_notes, prev_tag):
        feature_str = []
        feature_str.append("prev_tag=%s"%prev_tag)
        for j, pn in enumerate(prev_notes):
            if pn[0] == "rest":
                note_num = "note[-%s]=rest"%(len(prev_notes)-j)
            else:
                note_num = "note[-%s]=note"%(len(prev_notes)-j)
            note_duration = "length[-%s]=%s"%(len(prev_notes)-j, pn[1])
            feature_str.append(note_num)
            feature_str.append(note_duration)
        for j, nn in enumerate(next_notes):
            if nn[0] == "rest":
                note_num = "note[%s]=rest"%j
            else:
                note_num = "note[%s]=note"%j
            note_duration = "length[%s]=%s"%(j, nn[1])
            feature_str.append(note_num)
            feature_str.append(note_duration)
        feature = [feature2idx[f] for f in feature_str]
        return feature

    def update_note_position(notes, note_position, word):
        """
        - Move notes for the syllable count of the generated word.
        - Note that rest does not move.
            - if num[0]=rest, position+=1
        """
        if word.startswith(("<BOL>", "<BOB>")):
            pass
        else:
            word_length = int(len([y for y in word.split("|")[-1].split("_") if y != 'ッ']))       
            if word_length != 0:
                step = 0
                width = 0
                for n in notes[note_position::]:
                    num = n[0]
                    width += 1
                    if num == "rest":
                        pass
                    else:
                        step += 1
                    if step == word_length:
                        break
                note_position += width
        if note_position+1 >= len(notes):
            return note_position
        if notes[note_position][0] == 'rest' :
            note_position += 1
        return note_position

    """ generate """
    accept = []                                                 # accepted lyrics
    max_sent = len(notes) - 1
    prob_forward = [[] for l in range(max_sent)]                # prob_forward[t] = [(old_path, old_note_positions, prob), ...]
    for t in range(max_sent):
        if t == 0:
            old_path = (word2idx["<BOB>|<null>"], )
            old_note_positions = (0, )
            old_generated = dict(zip(old_note_positions, old_path))

            lengths = torch.Tensor([1]).long().to(device)

            # 1. Make input word vector
            x_word = torch.Tensor([[old_path[0]]]).long().to(device)

            # 2. Make input melody vector
            x_midi = np.zeros((1, t+1, feature_size))
            i = t+1
            prev_i = i - feature_window + 1
            if prev_i < 0:
                prev_i = 0
            next_i = i + feature_window
            if next_i > len(notes):
                next_i = len(notes)
            prev_notes = notes[prev_i:i]
            next_notes = notes[i:next_i]
            feature = make_feature(prev_notes, next_notes, "<BOB>")
            for f in feature:
                x_midi[0, 0, f] = 1
            x_midi = torch.Tensor(x_midi).to(device)

            # 3. Generate word
            syllable_output, lyrics_output, hidden = model(x_word, x_midi, lengths+1, hidden)
            dist = F.softmax(lyrics_output, dim=1).cpu().numpy()[0]
            dist[word2idx["<unk>"]] = 0.0
            stack = set()
            for s in range(100*window):
                new_index = sample(dist, temperature)
                new_word = idx2word[str(new_index)]
                new_path = old_path + (new_index, )
                new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                prob = math.log(dist[new_index])
                stack.add((new_path, new_note_positions, prob))
                if len(stack) >= window:
                    break
            for new in stack:
                prob_forward[t].append(new)
        else:
            x_word = np.zeros((window, t+1), dtype="int32")
            x_midi = np.zeros((window, t+1, feature_size))
            for w, (old_path, old_note_positions, old_prob) in enumerate(prob_forward[t-1]):
                # 1. Make input word vector
                for old_t, old_index in enumerate(old_path):
                    x_word[w, old_t] = old_index

                # 2. Make input melody vector
                if old_index == bob:
                    prev_tag = "<BOB>"
                elif old_index == bol:
                    prev_tag = "<BOL>"
                else:
                    prev_tag = "<WORD>"
                for old_t, old_midi_position in enumerate(old_note_positions):
                    i = old_midi_position
                    prev_i = i - feature_window + 1
                    if prev_i < 0:
                        prev_i = 0
                    next_i = i + feature_window
                    if next_i > len(notes):
                        next_i = len(notes)
                    prev_notes = notes[prev_i:i]
                    next_notes = notes[i:next_i]
                    feature = make_feature(prev_notes, next_notes, prev_tag)
                    for f in feature:
                        x_midi[w, old_t, f] = 1
            x_word = torch.Tensor(x_word).long().to(device)
            x_midi = torch.Tensor(x_midi).to(device)
            lengths = torch.Tensor([t+1]*window).long().to(device)

            # 3. Generate word
            hidden = model.init_hidden(20)
            syllable_output, lyrics_output, hidden = model(x_word, x_midi, lengths+1, hidden)
            lyrics_output = lyrics_output[-window::]            # extract last output
            dists = F.softmax(lyrics_output, dim=1).cpu().numpy()
            stack = set()
            for w in range(len(prob_forward[t-1])):
                dist = dists[w]
                dist[word2idx["<unk>"]] = 0.0
                old_path = prob_forward[t-1][w][0]
                if old_path[-1] in (bob, bol):
                    was_segment = True
                else:
                    was_segment = False
                old_note_positions = prob_forward[t-1][w][1]
                old_prob = prob_forward[t-1][w][2]
                old_generated = dict(zip(old_note_positions, old_path))
                generate_bol = False
                generate_bob = False

                temp_stack = set()
                for s in range(100*window):
                    new_index = sample(dist, temperature)
                    if was_segment == True and new_index in (bob, bol):
                        continue
                    if new_index == bol:
                        generate_bol = True
                    if new_index == bob:
                        generate_bob = True
                    new_word = idx2word[str(new_index)]
                    new_path = old_path + (new_index, )
                    new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                    new_prob = math.log(dist[new_index]) + old_prob
                    temp_stack.add((new_path, new_note_positions, new_prob))
                    if len(temp_stack) >= window:
                        break

                if was_segment == False:
                    if generate_bol == True and generate_bob == False:
                        new_word = '<BOB>|<null>'
                        new_index = word2idx[new_word]
                        new_path = old_path + (new_index, )
                        new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                        new_prob = math.log(dist[new_index]) + old_prob
                        temp_stack.add((new_path, new_note_positions, new_prob))
                    elif generate_bol == False and generate_bob == True:
                        new_word = '<BOL>|<null>'
                        new_index = word2idx[new_word]
                        new_path = old_path + (new_index, )
                        new_note_positions = old_note_positions + (update_note_position(notes, old_note_positions[-1], new_word), )
                        new_prob = math.log(dist[new_index]) + old_prob
                        temp_stack.add((new_path, new_note_positions, new_prob))
                for item in temp_stack:
                    stack.add(item)

            count = 0
            for path, note_positions, prob in sorted(list(stack), key=lambda x:x[2], reverse=True):
                if note_positions[-1] >= max_sent:                  # accepted lyrics are saved
                    entropy = -prob/(len(path)-1)
                    accept.append((path, note_positions, entropy))
                elif note_positions[-1] < max_sent:
                    prob_forward[t].append((path, note_positions, prob))
                    count += 1
                if count >= window:
                    break
            if len(stack) > 0:
                last_max_position = max([Y[1][-1] for Y in list(stack)])
                prgs_num = 100 * last_max_position / len(notes)
                prgs_bar = "Generate lyrics [" + "=" * int(prgs_num/5) + ">" + "-" * (20 - int(prgs_num/5)) + "]"
                sys.stderr.write("\r%s (%.1f %%)"%(prgs_bar, prgs_num))

        # 4. Break
        if len(accept) >= 10:
            sys.stderr.write("\n")
            break

    # 5. Output
    path, note_positions, score = max(accept, key=lambda x:x[2])
    generated = [idx2word[str(idx)] for idx in path[1::]]
    return generated, note_positions, score


def save_lyrics(generated, notes, output_dir):
    yomi = []
    yomi_w_word = []
    line = []
    out_file = open(output_dir.rstrip("/")+ '/output.txt', 'w')
    lyrics_file = open(output_dir.rstrip("/") + '/output.lyric', 'w')
    word_idx = 0
    bob_idxs = set()
    bob_idxs.add(0)
    bol_idxs = set()
    for word in generated:
        if word == '<BOL>|<null>':
            lyrics_file.write("".join([w.split(",")[0] for w in line]) + '\n')
            out_file.write(" ".join(line) + '\n')
            line = []
            bol_idxs.add(word_idx)
        elif word == '<BOB>|<null>':
            lyrics_file.write("".join([w.split(",")[0] for w in line]) + '\n\n')
            out_file.write(" ".join(line) + '\n\n')
            line = []
            bob_idxs.add(word_idx)
        else:
            line.append(word)
            yomi += [jaconv.kata2hira(y) for y in word.split("|")[-1].split("_") if y != 'ッ']
            yomi_w_word += [(jaconv.kata2hira(y), word_idx, word) for y in word.split("|")[-1].split("_") if y != 'ッ']
            word_idx += 1
    if len(line) > 0:
        lyrics_file.write("".join([w.split(",")[0] for w in line]) + '\n')
        out_file.write(" ".join(line) + '\n')

    # Save JSON
    midi_json = {u'tracks': 1, u'resolution': 480, u'format': 1, u'stream': []}
    stack_rest = []
    yomi_position = 0
    for num, length in notes:
        length = length = int(length)
        if yomi_position >= len(yomi):
            break
        if num == 'rest':
            stack_rest.append(int(length))
        else:
            if len(stack_rest) > 0:
                midi_json[u'stream'].append({u'velocity': 64, 
                                             u'tick': int(sum(stack_rest)), 
                                             u'sub_type': u'noteOn', 
                                             u'channel': 0, 
                                             u'note_num': int(num)})
                midi_json[u'stream'].append({u'velocity': 64, 
                                             u'tick': int(length), 
                                             u'sub_type': u'noteOff', 
                                             u'channel': 0, 
                                             u'note_num': int(num), 
                                             u'lyrics':yomi[yomi_position]})
                stack_rest = []
            else:
                midi_json[u'stream'].append({u'velocity': 64, 
                                             u'tick': 0, 
                                             u'sub_type': u'noteOn', 
                                             u'channel': 0, 
                                             u'note_num': int(num)})
                midi_json[u'stream'].append({u'velocity': 64, 
                                             u'tick': int(length), 
                                             u'sub_type': u'noteOff', 
                                             u'channel': 0, 
                                             u'note_num': int(num), 
                                             u'lyrics':yomi[yomi_position]})
            yomi_position += 1
    json_file = open(output_dir.rstrip("") + "/output.json", 'w')
    json_file.write(json.dumps(midi_json, ensure_ascii=False))

    # Save readable format
    song = {'lyrics':[]}
    yomi_position = 0
    temp_lyrics = []
    for note_idx, (num, length) in enumerate(notes):
        length = int(length)
        if yomi_position >= len(yomi):
            break
        if num == 'rest':
            temp_lyrics.append([note_idx, num, length])
        else:
            word_yomi = jaconv.hira2kata(yomi_w_word[yomi_position][0])
            word_idx = yomi_w_word[yomi_position][1]
            word = yomi_w_word[yomi_position][2]
            sur, pos, dtl = word.split("|")[0].split(",")
            yomis = word.split("|")[1].split("_")
            len_yomi = len([y for y in yomis if y != "ッ"])
            accs = "<None>"
            if  word_idx in bob_idxs:
                temp_lyrics.append([note_idx, num, length, word_yomi, [sur, pos, dtl, yomis, accs], word_idx, "<BOB>"])
            elif  word_idx in bol_idxs:
                temp_lyrics.append([note_idx, num, length, word_yomi, [sur, pos, dtl, yomis, accs], word_idx, "<BOL>"])
            else:
                temp_lyrics.append([note_idx, num, length, word_yomi, [sur, pos, dtl, yomis, accs], word_idx, "<None>"])
            yomi_position += 1
    last_note_idx = len(temp_lyrics)-1
    for i, note in enumerate(temp_lyrics):
        if note[1] == "rest":
            if i == 0 or i == last_note_idx:
                song["lyrics"].append([note[0], note[1], note[2], "<None>", "<None>", "<None>", "<None>"])
            else:
                if temp_lyrics[i-1][-2] == temp_lyrics[i+1][-2]:
                    song["lyrics"].append(note[::] + ["<None>"] + temp_lyrics[i+1][-3::])
                else:
                    song["lyrics"].append([note[0], note[1], note[2], "<None>", "<None>", "<None>", "<None>"])
        else:
            song["lyrics"].append(note[::])
    song["lyrics"].append([last_note_idx+1, "rest", "7680", "<None>", "<None>", "<None>", "<None>"])
    corpus_file = open(output_dir.rstrip("/") + "/output.readable", 'w')
    for note in song["lyrics"]:
        corpus_file.write(json.dumps(note, ensure_ascii=False) + "\n")


def main(args):

    argparse_dict = vars(args)
    print("------ Parameters -----")
    for k, v in argparse_dict.items():
        print("{:>16}:  {}".format(k, v))

    notes = convert(args.midi)
    with torch.no_grad():
        lyrics, positions, score = generate(notes=notes, 
                                            param=args.param, checkpoint=args.checkpoint, 
                                            seed=args.seed, window=args.window, 
                                            temperature=args.temperature)
    save_lyrics(lyrics, notes, args.output)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """ Data parameter """
    parser.add_argument("-midi", "--midi", dest="midi", default="./sample_data/sample.midi", type=str, help="MIDI file")
    parser.add_argument("-output", "--output", dest="output", default="./output/", type=str, help="Output directory")

    """ Model parameter """
    parser.add_argument("-param", "--param", dest="param", default="./checkpoint/model.param.json", type=str, help="Parameter file path")
    parser.add_argument("-checkpoint", "--checkpoint", dest="checkpoint", default="./checkpoint/model_05.pt", type=str, help="Checkpoint file path")

    """ Generation parameter """
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="Seed number for random library")
    parser.add_argument("-window", "--window", dest="window", default=20, type=int, help="Window size for beam search")
    parser.add_argument("-temperature", "--temperature", dest="temperature", default=1.0, type=float, help="Word sampling temperature")
    args = parser.parse_args()
    main(args)
