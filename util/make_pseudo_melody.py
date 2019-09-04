# -*- coding: utf-8 -*-
import sys
import re
import json
import argparse
from collections import defaultdict
import numpy as np
np.random.seed(0)
import random
random.seed(0)

import romkan
import jctconv

import re
en_p = re.compile(r"[a-zA-Z\']")
num_p = re.compile(r"[0-9\']")
hira_p = re.compile("[ぁ-ん]")
kata_p = re.compile("[ァ-ン]")
re_katakana = re.compile(r'^(?:[\u30A1-\u30F4|ー|_])+$')

import MeCab
import CaboCha
cabocha_tagger = CaboCha.Parser("-f1 -d ./dic/ipadic")
accent_tagger = MeCab.Tagger("-d ./dic/unidic")

import nltk
from nltk.stem.wordnet import WordNetLemmatizer as WNL
wnl = WNL()

en2kana = {}
for strm in open("./dic/en2kana.txt", "r"):
    en = strm.strip().split("\t")[0]
    kana = strm.strip().split("\t")[1]
    en2kana[en] = kana

"""
lyrics parsing functions
"""
def parse(text):
    phrase = []
    words = []
    for morph in cabocha_tagger.parseToString(text).split("\n"):
        if morph.strip() == "" or morph.strip() == "EOS": continue
        if morph.strip().split(" ")[0] == "*":
            if phrase:
                phrase_lyrics = get_phrase(phrase, phrase_info)
                phrase = []
                for word in phrase_lyrics['word']:
                    accent = []
                    for syllable in word['syllable']:
                        accent.append(syllable['accent'])
                    yomi = "_".join(get_yomi(word['info'].split(',')[-1]))
                    if yomi == "*":
                        yomi = "_".join(get_yomi(jctconv.hira2kata(word['sur'])))
                    words.append("%s,%s,%s"%(','.join(word['info'].split(',')[:-1:]), yomi, '_'.join(accent)))
            phrase_info = morph.split(" ")[1::]
        else:
            phrase.append(morph)
    phrase_lyrics = get_phrase(phrase, phrase_info)
    for word in phrase_lyrics['word']:
        accent = []
        for syllable in word['syllable']:
            accent.append(syllable['accent'])
        yomi = "_".join(get_yomi(word['info'].split(',')[-1]))
        if yomi == "*":
            yomi = "_".join(get_yomi(jctconv.hira2kata(word['sur'])))

        words.append("%s,%s,%s"%(','.join(word['info'].split(',')[:-1:]), yomi, '_'.join(accent)))
    return words

def get_phrase(phrase, phrase_info):
    phrase_lyrics = {"sur": " ".join([w.split("\t")[0] for w in phrase]), 
            "info": " ".join(phrase_info), 
            "word":[]}
    accent = get_accent("".join([w.split("\t")[0] for w in phrase]))
    mora = 0
    acc_idx = 0
    for word in phrase:
        sur = word.split("\t")[0]
        if en_p.match(sur):
            word = parse_eng_word(sur)
        kana = word.split("\t")[1].split(",")[-1]
        kana = "".join(get_yomi(kana))
        if kana == "*":
            kana = jctconv.hira2kata(sur)
        mora += get_mora(kana)
        kana = "q".join(kana.split("ッ"))
        kana = "".join(kana.split("ー"))
        kana = "N".join(kana.split("ン"))
        word_alpha = split_alpha(kana)
        word_lyrics = {"sur":sur, "kana":kana, "info":word, "syllable":[]}
        for w_a in word_alpha:
            syllable_lyrics = {}
            out_char = [w_a[0], w_a[1]]
            if out_char[1] == "*":
                syllable_lyrics["sur"] = w_a[0]
                syllable_lyrics["roma"] = w_a[1]
                syllable_lyrics["accent"] = "*"
            else:
                if len(accent) <= mora:      
                    syllable_lyrics["sur"] = w_a[0]
                    syllable_lyrics["roma"] = w_a[1]
                    syllable_lyrics["accent"] = accent[-1]
                else:
                    syllable_lyrics["sur"] = w_a[0]
                    syllable_lyrics["roma"] = w_a[1]
                    if acc_idx+1 > len(accent):
                        syllable_lyrics["accent"] = accent[-1]
                    else:
                        syllable_lyrics["accent"] = accent[acc_idx]
                acc_idx += 1
            word_lyrics["syllable"].append(syllable_lyrics)
        phrase_lyrics["word"].append(word_lyrics)
    return phrase_lyrics

def parse_eng_word(term):
    sur = nltk.word_tokenize(term.strip())[0]
    pos =  nltk.pos_tag(sur)[0][1]
    lemma = wnl.lemmatize(sur.lower())
    pos2 = "*"
    pos3 = "*"
    pos4 = "*"
    form1 = "*"
    form2 = "*"
    base = lemma
    kana = en2kana.get(sur.lower(), sur)
    yomi = kana
    return "%s\t%s,%s,%s,%s,%s,%s,%s,%s,%s"%(sur, pos, pos2, pos3, pos4, form1, form2, base, kana, yomi)

def split_alpha(line):
    out = []
    for char in line:
        if char in ('ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ'):
            if out:
                out[-1] += char
            else:
                out.append(char)
        else:
            out.append(char)
    return [("ッ".join("ン".join(kana.split("N")).split("q")), romkan.to_roma(kana)) for kana in out]

def get_mora(kana):
    mora = len(kana)
    for char in kana:
        if char in ('ァ', 'ィ', 'ゥ', 'ェ', 'ォ', 'ャ', 'ュ', 'ョ'):
            mora -= 1
    if mora < 0:
        mora = 0
    return mora

def get_yomi(kana):
    yomi = []
    for char in kana:
        if char == 'ー':
            if len(yomi) == 0: continue
            if yomi[-1] in ('ア','カ','ガ','サ','ザ','タ','ダ','ナ','ハ','バ','パ','マ','ヤ','ラ','ワ','キャ','ギャ','シャ','ジャ','チャ','ヂャ','ニャ','ヒャ','ビャ','ピャ','ミャ','リャ','ファ','ヴァ'):
                yomi.append("ア")
            elif yomi[-1] in ('イ','キ','ギ','シ','ジ','チ','ヂ','ニ','ヒ','ビ','ピ','ミ','リ','ヰ','フィ','ヴィ','ディ','ティ', 'ウィ'):
                yomi.append("イ")
            elif yomi[-1] in ('ウ','ク','グ','ス','ズ','ツ','ヅ','ヌ','フ','ブ','プ','ム','ユ','ル','キュ','ギュ','シュ','ジュ','チュ','ヂュ','ニュ','ヒュ','ビュ','ピュ','ミュ','リュ','フュ','ヴュ', 'トゥ', 'ドゥ'):
                yomi.append("ウ")
            elif yomi[-1] in ('エ','ケ','ゲ','セ','ゼ','テ','デ','ネ','ヘ','ベ','ペ','メ','レ','フェ','ウェ','ヴェ','シェ','ジェ','チェ','ヂェ','ニェ','ヒェ','ビェ','ピェ','ミェ', 'イェ'):
                yomi.append("エ")
            elif yomi[-1] in ('オ','コ','ゴ','ソ','ゾ','ト','ド','ノ','ホ','ボ','ポ','モ','ヨ','ロ','ヲ','キョ','ギョ','ショ','ジョ','チョ','ヂョ','ニョ','ヒョ','ビョ','ピョ','ミョ','リョ','フォ','ブォ','ウォ','ヴォ'):
                yomi.append("オ")
            elif yomi[-1] == "ン":
                yomi.append("ン")
        elif char in ('ァィゥェォャュョ'):
            if len(yomi) > 0:
                yomi[-1] = yomi[-1] + char
            else:
                yomi.append(char)
        else:
            yomi.append(char)
    return yomi

""" 
For extracting Japanese accect information,  we used UniDic (http://unidic.ninjal.ac.jp/).
However, it is very difficult to analyze UniDic accent information (see https://repository.dl.itc.u-tokyo.ac.jp/?action=repository_action_common_download&item_id=3407&item_no=1&attribute_id=14&file_no=1 Chapter2).
We inplemented accent parser below accoding to this paper.
"""
def get_accent(phrase):
    accent_info = []
    out_accent = 0
    out_mora = 0
    head_f = False
    for morph in accent_tagger.parse(phrase).split('\n'):
        if len(morph.split("\t")) <= 8:break        
        if morph == "EOS" or morph == "": continue
        mora = get_mora(morph.split("\t")[1])
        pos = morph.split("\t")[4].split("-")[0]
        if morph.split("\t")[7] == "":
            acc_position = 0        
        else:
            acc_position = int(morph.split("\t")[7].split(",")[0])
        if morph.split("\t")[8] == "":
            acc_joint = None
        else:
            acc_joint = morph.split("\t")[8]
        if morph.split("\t")[9] != "":
            acc_metric = morph.split("\t")[9]
            M, back_m = acc_metric.split("@")
            if M == "M1":
                acc_position = mora - int(back_m)
            if M == "M2":
                if acc_position == 0:
                    acc_position = mora - int(back_m)
            if M == "M4":
                if acc_position == 0:
                    pass
                elif acc_position == 1:
                    pass
                else:
                    acc_position = mora - int(back_m)
        else:
            acc_metric = None
        if acc_joint:
            if acc_joint[0] == "P":
                head_f = True
            else:
                if len(accent_info) == 0:   
                    out_accent = acc_position
                else:   
                    prev_acc_joint = accent_info[-1][2]
                    prev_pos = accent_info[-1][4]
                    # step1.3: 接頭辞アクセント結合規則
                    if head_f == True and "名詞" in pos:
                        if prev_acc_joint == "P1":       # 一体化型
                            if acc_position == 0:
                                out_accent = 0
                            else:
                                out_accent = out_mora + acc_position
                        elif prev_acc_joint == "P2":       # 自立語結合型
                            if acc_position == 0:
                                out_accent =  out_mora + 1
                            else:
                                out_accent = out_mora + acc_position
                        elif prev_acc_joint == "P4":       # 混合型
                            if acc_position == 0:
                                out_accent =  out_mora + 1
                            else:
                                out_accent = out_mora + acc_position
                        elif prev_acc_joint == "P6":       # 平板型
                            out_accent = 0
                        head_f = False
                    elif len(acc_joint.split(",")[0].split("%")) > 1:
                        acc_joints = acc_joint.split(",")
                        for j in range(len(acc_joints)):
                            sub_rule = acc_joints[j]
                            if j+1 < len(acc_joints):
                                if len(acc_joints[j].split("%")) == 1:    # F6のようなフォーマットの時
                                    continue
                                if len(acc_joints[j+1].split("%")) == 1:    # F6のようなフォーマットの時
                                    sub_rule = acc_joints[j] + "," + acc_joints[j+1]
                            if len(sub_rule.split("%")) == 1: continue
                            sub_rule_pos = sub_rule.split("%")[0]
                            sub_rule_F = sub_rule.split("%")[1]
                            if sub_rule_pos in prev_pos:
                                F = sub_rule_F.split("@")[0]
                                if F == "F1":       # 従属型(そのまま)
                                    pass
                                elif F == "F2":     # 不完全支配型
                                    joint_value = int(sub_rule_F.split("@")[1].split(",")[0][:2])
                                    if out_accent == 0:  # 0型アクセントに接続した場合は結合アクセント価を足す
                                        out_accent = out_mora +  joint_value
                                elif F == "F3":     # 融合型
                                    joint_value = int(sub_rule_F.split("@")[1].split(",")[0][:2])
                                    if out_accent != 0:  # 0型アクセント以外に接続した場合は結合アクセント価を足す
                                        out_accent = out_mora + joint_value
                                elif F == "F4":     # 支配型 とにかく結合アクセント価を足す
                                    joint_value = int(sub_rule_F.split("@")[1].split(",")[0][:2])
                                    out_accent = out_mora + joint_value
                                elif F == "F5":     # 平板型 アクセントが消失する
                                    out_accent = 0
                                elif F == "F6": 
                                    joint_value1 = int(sub_rule_F.split("@")[1].split(",")[0][:2])
                                    joint_value2 = int(sub_rule_F.split("@")[1].split(",")[1][:2])
                                    if out_accent == 0:  # 0型アクセントに接続した場合は第一結合アクセント価を足す
                                        out_accent = out_mora + joint_value1
                                    else:  # 0型アクセント以外に接続した場合は第二結合アクセント価を足す
                                        out_accent = out_mora + joint_value2
                                break
                    elif acc_joint[0] == "C":
                        if acc_joint == "C1":       # 自立語結合保存型
                            out_accent = out_mora + acc_position
                        if acc_joint == "C2":       # 自立語結合生起型
                            out_accent = out_mora + 1
                        if acc_joint == "C3":       # 接辞結合標準型
                            out_accent = out_mora
                        if acc_joint == "C4":       # 接辞結合平板化型
                            out_accent = 0
                        if acc_joint == "C5":       # 従属型
                            out_accent = out_accent
                        if acc_joint == "C10":       # その他
                            out_accent = out_accent
        else:
            if len(accent_info) == 0:
                out_accent = acc_position

        
        accent_info.append((mora, acc_position, acc_joint, acc_metric, pos))
        out_mora += mora
    hl = []
    if out_accent == 0: #0板
        hl.append("L")
        for i in range(out_mora-1):
            hl.append("H")
    elif out_accent == 1: #1型
        hl.append("H")
        for i in range(out_mora-1):
            hl.append("L")
    else:       # それ以外
        hl.append("L")
        for i in range(out_accent-1):
            hl.append("H")
        for i in range(out_mora-out_accent):
            hl.append("L")
    return hl

def get_long_sound(kana):
    if kana in ('ア','カ','ガ','サ','ザ','タ','ダ','ナ','ハ','バ','パ','マ','ヤ','ラ','ワ','キャ','ギャ','シャ','ジャ','チャ','ヂャ','ニャ','ヒャ','ビャ','ピャ','ミャ','リャ','ファ','ヴァ'):
        return "ア"
    elif kana in ('イ','キ','ギ','シ','ジ','チ','ヂ','ニ','ヒ','ビ','ピ','ミ','リ','ヰ','フィ','ヴィ','ディ','ティ', 'ウィ'):
        return "イ"
    elif kana in ('ウ','ク','グ','ス','ズ','ツ','ヅ','ヌ','フ','ブ','プ','ム','ユ','ル','キュ','ギュ','シュ','ジュ','チュ','ヂュ','ニュ','ヒュ','ビュ','ピュ','ミュ','リュ','フュ','ヴュ', 'トゥ', 'ドゥ'):
        return "ウ"
    elif kana in ('エ','ケ','ゲ','セ','ゼ','テ','デ','ネ','ヘ','ベ','ペ','メ','レ','フェ','ウェ','ヴェ','シェ','ジェ','チェ','ヂェ','ニェ','ヒェ','ビェ','ピェ','ミェ', 'イェ'):
        return "エ"
    elif kana in ('オ','コ','ゴ','ソ','ゾ','ト','ド','ノ','ホ','ボ','ポ','モ','ヨ','ロ','ヲ','キョ','ギョ','ショ','ジョ','チョ','ヂョ','ニョ','ヒョ','ビョ','ピョ','ミョ','リョ','フォ','ブォ','ヴォ','ウォ'):
        return "オ"
    elif kana == "ン":
        return "ン"
    else:
        exit(1)


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

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class Generator:
    def __init__(self, model):
        self.seg_type_dist = defaultdict(lambda: defaultdict(float))
        self.seg_length_dist = defaultdict(lambda: defaultdict(float))
        self.hight_tri_dist = defaultdict(lambda: defaultdict(lambda: 0.000001))
        self.hight_set = set()
        for strm in open(model, "r"):
            data = json.loads(strm.strip())
            if data["type"] == "seg_type":
                self.seg_type_dist[data["seg_type"]][data["note_type"]] = data["prob"]
            elif data["type"] == "seg_length":
                self.seg_length_dist[data["seg_type"]][data["note_length"]] = data["prob"]
            elif data["type"] == "hight_tri":
                self.hight_tri_dist[tuple(data["prev_hight"])][data["note_hight"]] = data["prob"]
                if data["note_hight"] != "rest":
                    self.hight_set.add(data["note_hight"])
        self.hight_set = list(self.hight_set)

    def generate(self, lyrics):
        syllables = []
        segment_types = []
        segment_type = "<None>"
        idx = -1
        wdx = 0
        idx2word = {}
        idx2end = {}
        idx2wdx = {}
        for line in lyrics:
            if line != "":
                if segment_type != "<BOB>":
                    segment_type = "<BOL>"
                    if len(segment_types) == 0:
                        segment_type = "<BOB>"
                for morph in parse(line):
                    info = morph.split("\t")
                    yomi = info[1].split(",")[-2].split("_")
                    for y in yomi:
                        idx += 1
                        if y == "ッ":
                            continue
                        segment_types.append(segment_type)
                        syllables.append((y, idx))
                        idx2word[idx] = morph
                        idx2end[idx] = False
                        idx2wdx[idx] = wdx
                        segment_type = "<None>"
                    idx2end[idx] = True
                    wdx += 1
            else:
                segment_type = "<BOB>"
        segment_types.append("<BOB>")

        queue = ["rest"]
        iidx = 1
        old_wdx = -1
        _seg_type = "<BOB>"
        generated_melody = []
        generated_melody.append([0, "rest", "7680", "<None>", "<None>", "<None>", "<None>"])
        for i, ((sy, idx), next_segment_type) in enumerate(zip(syllables, segment_types[1::])):
            segment_type = segment_types[i]
            # STEP1: generate note and duration
            generated_length_idx = sample(list(self.seg_length_dist[next_segment_type].values()))
            generated_length = list(self.seg_length_dist[next_segment_type].keys())[generated_length_idx]
            if len(queue) == 1:
                generated_hight_idx = sample(list(self.hight_tri_dist.get((queue[-1], )).values()))
                generated_hight = list(self.hight_tri_dist.get((queue[-1], )).keys())[generated_hight_idx]
            else:
                if (queue[-2], queue[-1]) in self.hight_tri_dist:
                    generated_hight_idx = sample(list(self.hight_tri_dist.get((queue[-2], queue[-1])).values()))
                    generated_hight = list(self.hight_tri_dist.get((queue[-2], queue[-1])).keys())[generated_hight_idx]
                else:
                    generated_hight = random.choice(self.hight_set)

            queue.append(generated_hight)
            morph = idx2word[idx]
            info = morph.split("\t")
            sur = info[0]
            pos = info[1].split(",")[0]
            dtl = info[1].split(",")[1]
            yomi = info[1].split(",")[-2].split("_")
            acc = info[1].split(",")[-1].split("_")

            wdx = idx2wdx[idx]
            if old_wdx != wdx:
                _seg_type = segment_type
            generated_melody.append([iidx, str(generated_hight), str(generated_length), sy, [sur, pos, dtl, yomi, acc], wdx, _seg_type])
            iidx += 1

            # STEP2: generate rest and duration
            if queue[-1] != "rest":
                generated_type_idx = sample(list(self.seg_type_dist[next_segment_type].values()))
                generated_type = list(self.seg_type_dist[next_segment_type].keys())[generated_type_idx]
                if generated_type == "rest":
                    rest_length_idx = sample(list(self.seg_length_dist[next_segment_type].values()))
                    rest_length = list(self.seg_length_dist[next_segment_type].keys())[generated_length_idx]
                    queue.append("rest")
                    if idx2end[idx] == True:
                        generated_melody.append([iidx, "rest", str(rest_length), "<None>", "<None>", "<None>", "<None>"])
                    else:
                        generated_melody.append([iidx, "rest", str(rest_length), "<None>", [sur, pos, dtl, yomi, acc], wdx, _seg_type])
                    iidx += 1

            old_wdx = wdx

        if generated_melody[-1][1] == "rest":
            generated_melody[-1][2] = "7680"
        else:
            last_idx = generated_melody[-1][0] + 1
            generated_melody.append([last_idx, "rest", "7680", "<None>", "<None>", "<None>", "<None>"])

        return generated_melody





def main(args):
    g = Generator(args.model)
    c = 1
    for strm in open(args.lyrics, "r"):
        sys.stderr.write("\rGenerate pseudo melody %s"%c)
        c += 1
        data = json.loads(strm.strip())
        gen_melody = g.generate(data["lyric"])
        print(json.dumps({"artist":data["artist"], "title":data["title"], "lyrics":gen_melody}, ensure_ascii=False))
    sys.stderr.write("\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", default="./model/probs.jsonl", type=str, help="model path")
    parser.add_argument("-l", "--lyrics", dest="lyrics", default="../sample_data/sample_lyrics.jsonl", type=str, help="lyrics path")
    args = parser.parse_args()
    main(args)

