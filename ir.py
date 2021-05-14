import nltk
import math
import sys
import os
from collections import Counter
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


query = {}
text = {}
closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]

def preprocess_sentence(sentence):
    porters = PorterStemmer()
    token = nltk.word_tokenize(sentence.strip(' .'))
    splitted = [w for w in token if not w in closed_class_stop_words]
    res = []
    for w in splitted:
        res.append(porters.stem(w))
    return res

def process_text(file, dictionary):
    f = open(file, "r")
    data = f.readlines()
    f.close()
    qid = 0
    temp = ''
    start = False
    for index, value in enumerate(data):
        line = value.strip('\n').split(' ')
        if (index + 1) < len(data):
            if line[0] == '.I':
                start = False
                dictionary[qid] = preprocess_sentence(temp)
                qid += 1
                temp = ''
            elif line[0] == '.W':
                start = True
            elif start:
                temp += value.replace(',', '').strip(' .\n') + ' '
        else:
            dictionary[qid] = preprocess_sentence(temp)

def calculate_IDF(doc):
    wlist = []
    for item in doc.values():
        wlist += item
    IDF = Counter(wlist)
    for word, freq in IDF.items():
        IDF[word] = math.log(((len(doc) - 1) / freq))
    TF_IDF = {}
    for index, item in doc.items():
        TF = Counter(item)
        for word, freq in TF.items():
            TF[word] = IDF[word] * (freq / len(TF))
        TF_IDF[index] = TF

    return TF_IDF

def getCosine(qinput, qid, text_input):
    cosine = {}
    text_IDF = calculate_IDF(text)
    query_IDF = calculate_IDF(query)
    for abstract_id, abstract in text_input.items():
        if abstract_id != 0:
            numerator, denominator1, denominator2 = 0, 0, 0
            for word in qinput:
                if word in abstract:
                    numerator += query_IDF[qid][word] * text_IDF[abstract_id][word]
                    denominator1 += pow(query_IDF[qid][word], 2)
                    denominator2 += pow(text_IDF[abstract_id][word], 2)
            if numerator != 0:
                score = numerator / math.sqrt(denominator1 * denominator2)
                line = str(qid) + ' ' + str(abstract_id) + ' ' + str(round(score,4))
                cosine[line] = round(score,4)

            else:
                line = str(qid) + ' ' + str(abstract_id) + ' ' + '0'
                cosine[line] = 0
    return cosine

def output():
    with open('output.txt', 'a') as file:
        for qid, q in query.items():
            if qid != 0:
                cosine = getCosine(q, qid, text)
                print('sorting')
                sort_cosine = {k: cosine[k] for k in sorted(cosine, key=cosine.get, reverse=True)}
                for i in sort_cosine.keys():
                    file.write(i)
                    file.write('\n')
                print('finished working on the ' + str(qid) + ' text')
    file.close()

def main():
    process_text('cran.qry', query) 
    process_text('cran.all.1400', text) 
    output()
