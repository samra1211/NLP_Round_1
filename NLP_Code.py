import re #importing regular expression operations
import string
import nltk #Natural Language ToolKit
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import operator
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt #For Graphs
from wordcloud import WordCloud

#Importing book in text format
f = open('NLP_jurafsky_martin.txt','r')
content = f.read()
#print(content)
f.close()

#Preprocessing 1 : removing numbers
content = re.sub(r'\d+', '', content)
#print(content)

#Preprocessing 2 : removing leading and trailing white spaces and new line characters
content = content.strip()
content = re.sub('\n', ' ', content)
#print(content)

#Preprocessing 3 : Removing chapter names by removing all capital words
def onlyUpper(word):
    for ch in word :
        if not ch.isupper():
            return False
    return True

content = ' '.join(w for w in nltk.wordpunct_tokenize(content) if not onlyUpper(w))
#print(content)

#Preprocessing 4 : converting the text to all lowercase letters
content = content.lower()
#print(content)
    
#Preprocessing 5 : removing punctuation
for w in content :
    if w in string.punctuation:
        content = content.replace(w, '')
content = content.replace("’", '')
content = content.replace('©', '')
content = content.replace('“', '')
content = content.replace('—', '')
content = content.replace('”', '')
#print(content)

#Preprocessing 6 : removing foreign words
words = set(nltk.corpus.words.words())
content = ' '.join(w for w in nltk.wordpunct_tokenize(content) if w in words or not w.isalpha())
#print(content)

#Tokenizing
tokens = word_tokenize(content)
#print(tokens)

#Preprocessing 7 : Removing one letter word
token1 = [w for w in tokens if len(w) > 1]
#print(token1)
content = ' '.join(w for w in token1)
#print(content)

#Frequency Distribution
word_freq = dict(Counter(token1).most_common(20))
#print(word_freq)
xaxis = word_freq.keys()
yaxis = word_freq.values()
fig = plt.figure(figsize = (20, 5))
plt.bar(xaxis, yaxis)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Frequency Distribution before removing stop words")
plt.show()

#Creating wordcloud before removing stop words
word_cloud = WordCloud(background_color = 'white', stopwords = None).generate(content)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("")
plt.show()

#Preprocessing 8 : Removing stop words
stop_words = set(stopwords.words('english'))
token2 = [i for i in token1 if not i in stop_words]
#print(token2)
content = ' '.join(w for w in token2)
#print (content)

#Frequency Distribution after removing stop words
word_freq = dict(Counter(token2).most_common(20))
#print(word_freq)
xaxis = word_freq.keys()
yaxis = word_freq.values()
fig = plt.figure(figsize = (20, 5))
plt.bar(xaxis, yaxis)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Frequency Distribution after removing stop words")
plt.show()

#Creating wordcloud after removing stop words
word_cloud = WordCloud(background_color = 'white').generate(content)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("")
plt.show()

#Word Length vs Frequency
def func(freq) :
    word_lengths = {}

    for i in freq.keys():
        if len(i) not in word_lengths.keys():
            word_lengths[len(i)] = freq[i]
        else:
            word_lengths[len(i)] += freq[i]
    x = []
    y = []
    for i in word_lengths.keys():
        x.append(i)
    x.sort()
    for i in x:
        y.append(word_lengths[i])
    plt.xlabel('Word Length')
    plt.ylabel('Frequency')
    plt.title('Plot of Word Length vs Frequency')
    fig = plt.plot(x, y, '-')
    plt.show()

#Before removing stopwords
freq1 = dict(Counter(token1))
func(freq1)

#After removing stopwords
freq2 = dict(Counter(token2))    
func(freq2)

#POS Tagging
tagged = nltk.pos_tag(token2) #default tagset is Penn Treebank Tagset
print(tagged)

tag_freq = {}
for tag in tagged :
    if not tag[1] in tag_freq.keys() :
        tag_freq[tag[1]] = 1
    else :
        tag_freq[tag[1]] += 1

sorted_freq = sorted(tag_freq, key = tag_freq.get, reverse = True)
del sorted_freq[20:]

x = []
y = []
for i in sorted_freq:
    x.append(i)
    y.append(tag_freq[i])

plt.xlabel('POS Tags')
plt.ylabel('Frequency')
plt.title('Plot of POS Tags and their frequencies')
plt.xticks(rotation = 'vertical')
fig = plt.plot(x, y, '-')
plt.show()

