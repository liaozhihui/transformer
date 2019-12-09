#coding:utf-8
from tqdm import tqdm
import jieba


                


with open('cmn.txt','r',encoding="utf-8") as f:
    data = f.readlines()
    inputs = []
    outputs = []
    
    for line in tqdm(data):
        [en,ch] = line.strip('\n').split('\t')
        inputs.append(en.replace(',',' ,')[:-1].lower())
        outputs.append(ch[:-1])
    
    inputs = [en.split(" ") for en in inputs]
    outputs = [[char for char in jieba.cut(line) if char != ' '] for line in tqdm(outputs)]

def get_vocab(data,init=['<PAD>']):
    vocab  = init
    for line in tqdm(data):
        for word in line:
            if word not in vocab:
                vocab.append(word)
    return vocab
   
SOURCE_CODES = ['<PAD>']
TARGET_CODES = ['<PAD>','<GO>','<EOS>']

encoder_vocab = get_vocab(inputs,init=SOURCE_CODES)
decoder_vocab = get_vocab(outputs,init=TARGET_CODES)
print(encoder_vocab[:10])
print(decoder_vocab[:10])

    
        
    