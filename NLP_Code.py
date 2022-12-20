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
f = open('/Users/Samra/Desktop/NLP_jurafsky_martin.txt','r')
data = f.read()
#print(content)
f.close()

def preprocess(content):
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
    #content = content.lower()
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
    return content, token1

content, token1 = preprocess(data)
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
#plt.show()

#Creating wordcloud before removing stop words
word_cloud = WordCloud(background_color = 'white', stopwords = None).generate(content)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("")
#plt.show()

#Preprocessing 8 : Removing stop words
stop_words = set(stopwords.words('english'))
#print(stop_words)
token2 = [i for i in token1 if i not in stop_words]
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
#plt.show()

#Creating wordcloud after removing stop words
word_cloud = WordCloud(background_color = 'white').generate(content)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.title("")
#plt.show()

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
    #plt.show()

#Before removing stopwords
freq1 = dict(Counter(token1))
func(freq1)

#After removing stopwords
freq2 = dict(Counter(token2))    
func(freq2)

#POS Tagging
tagged = nltk.pos_tag(token2) #default tagset is Penn Treebank Tagset
#print(tagged)

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
#plt.show()

                            ##  PROJECT ROUND 2  ##
#Finding Nouns and Verbs
def NounsVerbs(tagged) :
    nouns = []
    verbs = []
    for word in tagged :
        if(word[1][0] == 'N'): nouns.append(word[0])
        if(word[1][0] == 'V'): verbs.append(word[0])
    return nouns, verbs
nouns, verbs = NounsVerbs(tagged)
#print(nouns)
#print(verbs)

#WordSense Disambiguation
from nltk.corpus import wordnet
def synset(words, name) :
    categories = {}
    for word in words :
        cat = []
        for synset in wordnet.synsets(word):
            if(name in synset.lexname()):
                cat.append(synset.lexname())
        if len(cat) > 0 : pos, lexical_name = cat[0].split('.')
        categories[lexical_name] = categories.get(lexical_name, 0) + 1
    return categories

def all_synsets(n, v) :
    nouns = synset(n, 'noun')
    verbs = synset(v, 'verb')
    return nouns, verbs

noun_syn, verb_syn = all_synsets(nouns, verbs)
#print(noun_syn)
#print(verb_syn)

#Plotting Frequencies
import seaborn as sns
def NounFrequencyPlot(nouns):
    g = sns.barplot(x=list(nouns.keys()), y=list(nouns.values()))
    g.set_xticklabels(g.get_xticklabels(), rotation = 90)
    g.set(xlabel = 'Noun Type', ylabel = 'Frequency', title = 'Noun Categorization')
    g.plot()
    plt.show()
    
def VerbFrequencyPlot(verbs):
    g = sns.barplot(x= list(verbs.keys()), y= list(verbs.values()))
    g.set_xticklabels(g.get_xticklabels(), rotation = 90)
    g.set(xlabel = 'Verb Type', ylabel = 'Frequency', title = 'Verb Categorization')
    g.plot()
    plt.show()

noun_syn, verb_syn = all_synsets(nouns, verbs)
#VerbFrequencyPlot(verb_syn)
#NounFrequencyPlot(noun_syn)

##  PART-2

pattern = 'NP: {<DT>?<JJ>*<NN>}'
cp = nltk.RegexpParser(pattern)
# for sentence in tagged_set:
cs = cp.parse(tagged)
print(cs)

#Named Entity Recognition
import spacy
from spacy import displacy
nlp=spacy.load("en_core_web_sm")
f = open('/Users/Samra/Desktop/Extracted_text.txt','r')
content1 = f.read()
content1, token3 = preprocess(content1)
#print(content1)
f.close()
doc = nlp(content1)
print(f"{str(len(doc.ents))} entities in the chosen text")
displacy.render(doc, jupyter = False, style = 'ent')

def entity_recognition(content):
    doc = nlp(content)
    person=[]
    org=[]
    location=[]
    for x in doc :
        if(x.ent_type_ == "PERSON") and x.text not in person:
            person.append(x.text)
        if(x.ent_type_ == "ORG") and x.text not in org:
            org.append(x.text)
        if(x.ent_type_ == "LOCATION") and x.text not in location:
            location.append(x.text)
    return person, org, location

person, org, location = entity_recognition(content1)
print(f"Number of person entities = {str(len(person))}")
print(f"Number of org entities = {str(len(org))}")
print(f"Number of location entities = {str(len(location))}")

#Calculating Frequency
test_data = 'Kiritchenko and Mohammad examined the performance of sentiment analysis systems on pairs of sentences that were identical except for containing either a common African American first name like Shaniqua or a common European American first name like Stephanie It had long been known that Alexander Hamilton John Jay and James Madison wrote the anonymously published Federalist papers in to persuade NewYork to ratify the UnitedStates Constitution'
actual_entities = ['PERSON', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'NORP', 'NORP', 'ORDINAL', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'NORP', 'NORP', 'ORDINAL', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'PERSON', 'PERSON', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'ORG', 'PRODUCT', 'O', 'O', 'O', 'ORG', 'O', 'O', 'O', 'ORG', 'O']
predicted_entities = ['PERSON', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'NORP', 'NORP', 'ORDINAL', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'NORP', 'NORP', 'ORDINAL', 'O', 'O', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'PERSON', 'PERSON', 'PERSON', 'O', 'O', 'PERSON', 'PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
c_m = confusion_matrix(actual_entities, predicted_entities)
h = sns.heatmap(c_m, annot=True, fmt="d")
h.set_xticklabels(h.get_xticklabels(), rotation = 90, fontsize = 7)
h.set(xlabel = 'Predicted Entities', ylabel = 'Actual Entities', title = 'Confusion Matrix')
h.plot()
#plt.show()
#print(classification_report(actual_entities, predicted_entities))

#Entity Relationship
from nltk.sem.relextract import extract_rels, rtuple
from nltk.chunk import ne_chunk
def relationBetweenEntities(sentences):
    tokenized_sentences = [word_tokenize(sentence) for sentence in
sentences]
    tagged_sentences = [nltk.tag.pos_tag(sentence) for sentence in
tokenized_sentences]
    P1 = re.compile(r'.*\bin\b(?!\b.+ing)')
    print('PERSON and ORGANISATION')
    for i, sent in enumerate(tagged_sentences):
        sent = nltk.chunk.ne_chunk(sent)
        rels = extract_rels('PER', 'ORG', sent, corpus='ace', pattern=P1,
window=10)
        for rel in rels: print(rtuple(rel))
    print('\n')
    P2 = re.compile(r'.*\bof\b.*')
    print('PERSON and GPE')
    for i, sent in enumerate(tagged_sentences):
        sent = ne_chunk(sent)
        rels = extract_rels('PER', 'GPE', sent, corpus='ace', pattern=P2,
window=10)
        for rel in rels: print(rtuple(rel))
    print('\n')
    P3 = re.compile(r'.*\band\b.*')
    print('PERSON to PERSON')
    for i, sent in enumerate(tagged_sentences):
        sent = ne_chunk(sent)
        rels = extract_rels('PER', 'PER', sent, corpus='ace', pattern=P3,
window=10)
        for rel in rels: print(rtuple(rel))
    print('\n')

sentences = ['The dialogue above is from an early natural language processing system that could carry on a limited conversation with a user by imitating the responses of a Rogerian psychotherapist.', 'The Kleene star means zero or more occurrences of the immediately previous character or regular expression.', 'The Brown corpus is million word collection of samples from written English texts from different genres.', 'The Switchboard corpus of American English telephone conversations between strangers was collected in the early confusion matrix for visualizing how well binary classification system performs against gold standard labels.', 'In this section we introduce tests for statistical significance for classifiers drawing especially on the work of Dror et al and Berg Kirkpatrick et al.', 'Berg and Mohammad examined the performance of sentiment analysis systems on pairs of sentences that were identical except for containing either common African American first name like Shaniqua or common European American first name like Stephanie.', 'Naive Bayes is generative model that makes the bag of words assumption position matter and the conditional independence assumption words are conditionally independent of each other given the class.', 'It had long been known that Alexander Hamilton John Jay and James Madison wrote the anonymously published Federalist papers in to persuade New York to ratify the United States Constitution.', 'Non parametric methods for computing statistical significance were used first in in the competition.', 'Chinchor et al and even earlier in speech recognition Machine translation models are trained ona parallel corpus sometimes called bitext text that appears in two or more languages.']
relationBetweenEntities(sentences)
