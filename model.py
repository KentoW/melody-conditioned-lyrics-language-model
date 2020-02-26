# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F




class MCLLM(nn.Module):
    def __init__(self, word_dim, melody_dim, syllable_size, word_size, feature_size):
        super(MCLLM, self).__init__()
        self.hidden_dim = word_dim + melody_dim
        """ word embedding """
        self.embedding = nn.Embedding(word_size, word_dim)
        """" melody vector """
        self.fc_melody = nn.Linear(feature_size, melody_dim)
        """ LSTM """
        self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        """ output """
        self.fc_out_lyrics = nn.Linear(self.hidden_dim, word_size)
        self.fc_out_syllable = nn.Linear(self.hidden_dim, syllable_size)
        """ util """
        self.relu = nn.ReLU(True)
        self.bn_lyrics = nn.BatchNorm1d(word_size)
        self.bn_syllable = nn.BatchNorm1d(syllable_size)

    def forward(self, lyrics, melody, lengths, hidden):
        lengths = lengths-1
        local_batch_size = lyrics.shape[0]
        """ word embedding """
        word_emb = self.embedding(lyrics)
        """ melody vector """
        melody_vec = self.relu(self.fc_melody(melody))
        """ input vector """
        input_vec = torch.cat((word_emb, melody_vec), dim=2)
        input_vec = pack_padded_sequence(input_vec, lengths, batch_first=True)
        """ RNN """
        output, hidden = self.rnn(input_vec, hidden)
        """ output """
        lyrics_output = self.fc_out_lyrics(output[0])
        syllable_output = self.fc_out_syllable(output[0])
        if local_batch_size > 1:
            lyrics_output = self.bn_lyrics(lyrics_output)
            syllable_output = self.bn_syllable(syllable_output)

        return syllable_output, lyrics_output, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(1, bsz, self.hidden_dim), weight.new_zeros(1, bsz, self.hidden_dim))
