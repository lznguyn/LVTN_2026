import json
import os
from collections import Counter
import re

class WordVocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def __len__(self):
        return len(self.word2idx)

    @classmethod
    def build_vocab(cls, csv_path, min_freq=3):
        import pandas as pd
        df = pd.read_csv(csv_path)
        counter = Counter()
        for report in df['report'].dropna():
            tokens = re.findall(r'\w+', str(report).lower())
            counter.update(tokens)
        
        vocab = cls()
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab.add_word(word)
        return vocab

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word}, f, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vocab = cls()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        vocab.idx = len(vocab.word2idx)
        return vocab
