from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset
from stanza.models.tokenization.data import DataLoader
import torch
from tqdm import tqdm
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
    # TODO: send back numpy arrays, perhaps with pin_memory=True or by changing the collate_fn
    # pin_memory=True seems to slow things down enormously
    new_sentences = [[(sentence[0][0],
                       sentence[0][2])] for sentence in sentences]
    return new_sentences

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

def process_bulk_sentence(sentence, chunk):
    if len(sentence) > 1:
        raise ValueError("Was not expecting to process more than one sentence per paragraph via this method")
    s = sentence[0]
    return [[s[0], [0] * len(s[0]), s[1], [c for c in chunk]]]

def combine_bulk_data(docs, config, vocab):
    dataset = TokenizerDataset(docs, config, vocab)
    dataset = TorchDataLoader(dataset, batch_size=None, num_workers=8)
    combined_data = []
    combined_sentences = []
    for data, sentences in tqdm(dataset):
        processed_data = [[(i, 0) for i in chunk]
                          for chunk in data]
        combined_data.extend(processed_data)
        processed_sentences = [process_bulk_sentence(sentence, chunk) for sentence, chunk in zip(sentences, data)]
        combined_sentences.extend(processed_sentences)

    return combined_data, combined_sentences

