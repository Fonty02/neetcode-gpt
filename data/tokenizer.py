from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens=list(corpus)
        result=[]
        for _ in range(num_merges):
            if len(tokens)<2:
                break
            d={}
            for i in range(len(tokens)-1):
                pair=(tokens[i],tokens[i+1])
                d[pair]=d.get(pair,0)+1
            if not d:
                break
            max_freq=max(d.values())
            candidates=sorted(p for p, c in d.items() if c == max_freq)
            print(candidates)
            best=candidates[0]
            result.append([best[0],best[1]])
            new_tokens=[]
            i=0
            while i<len(tokens):
                if i<len(tokens)-1 and tokens[i]==best[0] and tokens[i+1]==best[1]:
                    new_tokens.append(best[0]+best[1])
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens=new_tokens
        return result
                
