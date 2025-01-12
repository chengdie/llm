import random
import torch
"""  test-1 -->
paragraphs = [
    ['this is a sentence', 'another sentence'],
    ['this is a paragraph', 'it has multiple sentences']
]

def tokenize(text, type = 'word'):
    if type == 'word':
        return [t.split() for t in text]
    elif type == 'char':
        return [list(t) for t in text]
    else:
        raise ValueError('type must be either word or char')

paragraphs = [tokenize(paragraph) for paragraph in paragraphs]
sentences = [sentence for paragraph in paragraphs
             for sentence in paragraph]
print(random.choice(random.choice(paragraphs)))
print(paragraphs)
print(sentences)
  <--  test-1"""


"""   test-2  extend&append -->  
examples = []
text1 = [(1,2,True),(2,3,False)]
text2 = [(4,6,True)]
# examples.append(text1)
# print(examples)
examples.extend(text2)
print(examples)
<-- test-2 """


