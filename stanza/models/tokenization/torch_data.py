from torch.utils.data import Dataset
from stanza.models.tokenization.data import DataLoader
import torch
import numpy as np

def condense_units(sentence):
    units = [w[0] for w in sentence]
    if all(len(unit) == 1 for unit in units):
        return "".join(units)
    else:
        return units


def condense_data(data):
    new_data = [condense_units(sentence) for sentence in data]
    return new_data

def condense_sentences(sentences):
    #new_sentences = [[(np.array(sentence[0][0]),
    #                   np.array(sentence[0][1]),
    #                   np.array(sentence[0][2]),
    #                   sentence[0][3])] for sentence in sentences]
    #return new_sentences
    return sentences

class TokenizerDataset(Dataset):
    def __init__(self, docs, config, vocab):
        self.docs = docs
        self.config = config
        self.vocab = vocab

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, index):
        # TODO: this completely ignores the alternate mechanism for Vietnamese
        doc = self.docs[index]
        text = doc.text

        loader = DataLoader(self.config, input_text=text, vocab=self.vocab, evaluation=True)
        data = condense_data(loader.data)
        sentences = condense_sentences(loader.sentences)
        return data, sentences

