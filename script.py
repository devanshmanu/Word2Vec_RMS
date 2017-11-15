import nltk
import numpy as np
from numpy import array
from nltk.corpus import brown
from gensim.models import Word2Vec


#Taking the percentage to make the subset of corpus
print("Initiated")
percentage = raw_input("Enter the percentage of subset wanted: ")

#Function to create subset of corpus
def select_elements(seq, perc):
	return seq[::int(100.0/int(perc))]

#Making a list from corpus(of a weird type)
corpusToList = list(brown.sents())

#Making the SubSet of the corpus
subsetList = select_elements(corpusToList,int(percentage))
#print(len(subsetList))


#overview output
print("Total size of corpus: ",len(corpusToList))
print("Total size of Subset: ", len(subsetList))

#Making the Word2Vec wodel of the given subset of corpus
model = Word2Vec(subsetList)
model_Orig = Word2Vec(corpusToList)
print("Choose from: ",subsetList)


W2V_word = raw_input("Which word? ")


#Final W2V
print("------")
print("New Model is")
print(model.most_similar(str(W2V_word)))
print("------")
print("Old Model Was")
print(model_Orig.most_similar(str(W2V_word)))
print("------")





#Finding the RMS
RMS_List_Subset=[]
for i in model.most_similar(str(W2V_word)):
	RMS_List_Subset.append((i[1]))
RMS_Subset =  np.sqrt(np.mean(array(RMS_List_Subset)**2))

RMS_List_Orig=[]
for i in model_Orig.most_similar(str(W2V_word)):
	RMS_List_Orig.append(i[1])
RMS_Orig = np.sqrt(np.mean(array(RMS_List_Orig)**2))

print("RMS Original: ", RMS_Orig)
print("RMS Subset: ", RMS_Subset)
diff = RMS_Orig - RMS_Subset
print("Difference: ", diff)




