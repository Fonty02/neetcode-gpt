import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        s=set()
        max_l=-1
        for sentence in positive:
            words=sentence.split(" ")
            max_l=max(max_l,len(words))
            for word in words:
                s.add(word)
        for sentence in negative:
            words=sentence.split(" ")
            max_l=max(max_l,len(words))
            for word in words:
                s.add(word)
        words=list(s)
        words.sort()
        word_to_idx = {word: i for i, word in enumerate(words, start=1)}
        n=len(positive)
        embedds = torch.zeros(2*n, max_l)
        for idx_s,sentence in enumerate(positive):
            words=sentence.split(" ")
            for idx_w,word in enumerate(words):
                embedds[idx_s][idx_w]=word_to_idx[word]
        for idx_s,sentence in enumerate(negative):
            words=sentence.split(" ")
            for idx_w,word in enumerate(words):
                embedds[idx_s+n][idx_w]=word_to_idx[word]
        return embedds

        """
        torch solution
        import torch
        import torch.nn as nn
        from typing import List

        class Solution:
            def get_dataset(self, positive: List[str], negative: List[str]) -> torch.Tensor:
                pos_tokenized = [sentence.split(" ") for sentence in positive]
                neg_tokenized = [sentence.split(" ") for sentence in negative]
                

                all_words = sorted(list({word for tokens in (pos_tokenized + neg_tokenized) for word in tokens}))
                word_to_idx = {word: i for i, word in enumerate(all_words, start=1)}
                

                pos_tensors = [torch.tensor([word_to_idx[w] for w in tokens]) for tokens in pos_tokenized]
                neg_tensors = [torch.tensor([word_to_idx[w] for w in tokens]) for tokens in neg_tokenized]
                

                all_tensors = pos_tensors + neg_tensors
                dataset = nn.utils.rnn.pad_sequence(all_tensors, batch_first=True, padding_value=0.0)
                return dataset.float()
        """


        
