from collections import Counter, OrderedDict

from stanza.models.common.vocab import BaseVocab, BaseMultiVocab, CharVocab
from stanza.models.common.vocab import VOCAB_PREFIX
from stanza.models.common.pretrain import PretrainedWordVocab


# class HierarchialBaseVocab:
#     """ A base class for common vocabulary operations. Each subclass should at least
#     implement its own build_vocab() function."""
#
#     def __init__(self, data=None, lang="", idx=0, tag_idx=0, cutoff=0, lower=False):
#         self.data = data
#         self.lang = lang
#         self.idx = idx
#         self.tag_idx = tag_idx
#         self.cutoff = cutoff
#         self.lower = lower
#         if data is not None:
#             self.build_vocab()
#         self.state_attrs = ['lang', 'idx', 'cutoff', 'lower', '_unit2id', '_id2unit']
#
#     def build_vocab(self):
#         raise NotImplementedError()
#
#     def state_dict(self):
#         """ Returns a dictionary containing all states that are necessary to recover
#         this vocab. Useful for serialization."""
#         state = OrderedDict()
#         for attr in self.state_attrs:
#             if hasattr(self, attr):
#                 state[attr] = getattr(self, attr)
#         return state
#
#     @classmethod
#     def load_state_dict(cls, state_dict):
#         """ Returns a new Vocab instance constructed from a state dict. """
#         new = cls()
#         for attr, value in state_dict.items():
#             setattr(new, attr, value)
#         return new
#
#     def normalize_unit(self, unit):
#         # be sure to look in subclasses for other normalization being done
#         # especially PretrainWordVocab
#         if unit is None:
#             return unit
#         if self.lower:
#             return unit.lower()
#         return unit
#
#     def unit2id(self, unit):
#         unit = self.normalize_unit(unit)
#         if unit in self._unit2id:
#             return self._unit2id[unit]
#         else:
#             return self._unit2id[UNK]
#
#     def id2unit(self, id):
#         return self._id2unit[id]
#
#     def map(self, units):
#         return [self.unit2id(x) for x in units]
#
#     def unmap(self, ids):
#         return [self.id2unit(x) for x in ids]
#
#     def __len__(self):
#         return len(self._id2unit)
#
#     def __getitem__(self, key):
#         if isinstance(key, str):
#             return self.unit2id(key)
#         elif isinstance(key, int) or isinstance(key, list):
#             return self.id2unit(key)
#         else:
#             raise TypeError("Vocab key must be one of str, list, or int")
#
#     def __contains__(self, key):
#         return self.normalize_unit(key) in self._unit2id
#
#     @property
#     def size(self):
#         return len(self)


class TagVocab(BaseVocab):
    """ A vocab for the output tag sequence. """

    def build_vocab(self):
        counter = Counter([w[self.idx][self.tag_idx]
                           for sent in self.data
                           for w in sent])
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: counter[k], reverse=True))
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}


class MultiVocab(BaseMultiVocab):
    def state_dict(self):
        """ Also save a vocab name to class name mapping in state dict. """
        state = OrderedDict()
        key2class = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
            key2class[k] = type(v).__name__
        state['_key2class'] = key2class
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        class_dict = {'CharVocab': CharVocab,
                      'PretrainedWordVocab': PretrainedWordVocab,
                      'TagVocab': TagVocab}
        new = cls()
        assert '_key2class' in state_dict, "Cannot find class name mapping in state dict!"
        key2class = state_dict.pop('_key2class')
        for k, v in state_dict.items():
            classname = key2class[k]
            new[k] = class_dict[classname].load_state_dict(v)
        return new


class CharVocab(BaseVocab):
    def build_vocab(self):
        if type(self.data[0][0]) is list:  # general data from DataLoader
            count_list = []
            for sent in self.data:
                for w in sent:
                    for tag in w[1]:
                        count_list.append((w[0], tag))
            counter = Counter(count_list)
            for k in list(counter.keys()):
                if counter[k] < self.cutoff:
                    del counter[k]
        else:  # special data from Char LM
            count_list = []
            for sent in self.data:
                for w in sent:
                    for tag in w[1]:
                        count_list.append((w[0], tag))
            counter = Counter(count_list)
        self._id2unit = VOCAB_PREFIX + list(sorted(list(counter.keys()), key=lambda k: (counter[k], k), reverse=True))
        self._unit2id = {w: i for i, w in enumerate(self._id2unit)}
