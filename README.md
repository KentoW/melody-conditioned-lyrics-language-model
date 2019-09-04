# Melody conditioned lyrics language model
Python scripts to train "melody conditioned lyrics language model" [[1]](#1) and generate *Japanese* lyrics. (We plan to implement an *English* version.)

## Requirement
### 1. Python and Python Packages
- `Python3.6`  
-  Machine Learning Packages
    - `torch==1.1.0` (if you use GPU, install CUDA enviroments)
    - `numpy==1.16.4`
-  Character Codes Packages
    - `jaconv`
    - `jctconv`
    - `romkan`
-  MIDI Packages
    - `mido`
-  Morpheme Parser Packages
    - `CaboCha` (manual install from this [link](https://taku910.github.io/cabocha/))
    - `mecab-python3`
    - `nltk` (you use WordNetLemmatizer)

### 2. Install Japanese parser
- You install... 
    - Japanese Morpheme Parser `Mecab` [url](https://taku910.github.io/mecab/)
    - Japanese Dependency Parser `CaboCha` [url](https://taku910.github.io/cabocha/)
    - python module for `MeCab` and `CaboCha`  
- Install Dictionary Files
```shell
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2013-06-20.zip 
unzip stanford-corenlp-full-2013-06-20.zip
```
Download `ipadic` and `unidic` from [MeCab: Yet Another Part-of-Speech and Morphological Analyzer](http://taku910.github.io/mecab/) and [UniDic](http://unidic.ninjal.ac.jp/download).  
```shell
mv unidic dic/
mv dic/dicrc dic/unidic/
mv ipadic dic/
```

## Usage
### 1. Make melody-lyrics alignment data (JSONL format)
- If you have UST file and lyrics text file, you can make alignment data by using the following [Python script](https://github.com/KentoW/melody-lyrics).
    - [jsonl file example](https://raw.githubusercontent.com/KentoW/melody-lyrics/master/src/data.jsonl)
- If you only have lyrics texts, you can create the alignment data automatically by using `util/make_pseudo_melody.py` in this repository.
    - Run `cd util` and `python make_pseudo_melody.py --lyrics ../sample_data/sample_lyrics.jsonl > ../sample_data/pseudo_data.jsonl`

### 2. Train melody conditioned lyrics language model
- Run `python train.py -data ./sample_data/pseudo_data.jsonl -checkpoint ./checkpoint/model`
- If you tune parameters, look help `python train.py -h`

### 3. Generate lyrics
- Run `python generate.py -midi ./sample_data/sample.midi -param ./checkpoint/model.param.json -checkpoint ./checkpoint/model_05.pt -output ./output`
- If you tune parameters, look help `python generate.py -h`

---

- <i id=1></i>[1] Kento Watanabe, Yuichiroh Matsubayashi, Satoru Fukayama, Masataka Goto, Kentaro Inui and Tomoyasu Nakano. A Melody-conditioned Lyrics Language Model. 
    In Proceedings of the 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2018)
