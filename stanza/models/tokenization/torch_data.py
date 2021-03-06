from torch.utils.data import Dataset
from stanza.models.tokenization.data import DataLoader
import torch


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

        d = DataLoader(self.config, input_text=text, vocab=self.vocab, evaluation=True)
        return d.data, d.sentences

