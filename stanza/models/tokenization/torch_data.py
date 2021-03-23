from torch.utils.data import Dataset
from stanza.models.tokenization.data import DataLoader
import torch


def condense_data(data):
    new_data = []
    for sentence in data:
        units = [w[0] for w in sentence]
        if all(len(unit) == 1 for unit in units):
            new_data.append("".join(units))
        else:
            new_data.append(units)
    return new_data


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
        return data, loader.sentences

