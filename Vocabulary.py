"""字典要负责的功能有把词转换为编号，把编号转换为词，允许往字典中添加词和删除词"""
class Vocabulary(object):
    def __init__(self, token_to_idx: dict = None):
        self.token_to_idx = token_to_idx if token_to_idx is not None else {}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

    def add_token(self, token: str):
        if token not in self.token_to_idx:
            index = len(self.token_to_idx)
            self.token_to_idx[token] = index
            self.idx_to_token[index] = token
        return self.token_to_idx[token]

    def add_tokens(self, tokens: list):
        for token in tokens:
            self.add_token(token)

    def get_index_by_token(self, token: str):
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        raise Exception("Token {0} cannot be found in Vocabulary.".format(token))

    def get_token_by_index(self, index: int):
        if len(self.idx_to_token) > index >= 0:
            return self.idx_to_token[index]
        return ""

    def __len__(self):
        return len(self.idx_to_token)

    def __str__(self):
        return "<Vocabulary(size={0})>".format(len(self))

"""
普通字典只需要提供查找单词的功能，完成的是单词和序号的映射。它可以复用到其它非序列的应用上。
序列字典需要提供的还有和序列相关的东西。
"""
class SequenceVocabulary(Vocabulary):
    def __init__(self, begin_token: str = "<BOS>", end_token: str = "<EOS>",
                 unk_token:str = "<UNK>", pad_token: str = "<PAD>"):
        super(SequenceVocabulary, self).__init__()
        self.begin_token = begin_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.pad_idx = self.add_token(pad_token)
        self.unk_idx = self.add_token(unk_token)
        self.begin_idx = self.add_token(begin_token)
        self.end_idx = self.add_token(end_token)

    def get_index_by_token(self, token: str):
        """
        override the function, if token is not in token_to_idx, just regard it as <UNK>.
        """
        return self.token_to_idx.get(token, 1)