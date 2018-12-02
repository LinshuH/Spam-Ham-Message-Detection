import pandas as pd
import numpy as np
# spam/not spam dataset
# https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
"""
Unigram split the sentence into one word each time for the evaluation
Bigrams split into two words
""" 

def generate_bigrams(word_list):
    """
    generate a list of bigram set: [{bigrams pair 1}, ...]
    each line: label text......
    word_list is text.....
    """
    bigrams = []
    for i in range (len(word_list)-1):
        a = word_list[i].lower()
        b = word_list[i+1].lower()
        pair = frozenset([a,b])
        bigrams.append(pair)
    return bigrams

def generate_corpus(alist):
    # ham: not spam;
    ham = {}
    spam = {}
    n_ham = 0
    n_spam = 0
    for line in alist:
        line = line.split()
        label = line[0]
        text = line[1:]
        biagram = generate_bigrams(text)
        
        # decompose text, GENERATE all the bigrams pairs
        if label == "ham":
            n_ham += 1
            for bia in biagram:
                if bia not in ham:
                    ham[bia] = 1
                else:
                    ham[bia] += 1
        else:
            n_spam += 1
            for bia in biagram:
                if bia not in spam:
                    spam[bia] = 1
                else:
                    spam[bia] += 1
    
    # find biagrams that exist in both spam and not spam for further calculation.
    # generate ham_corpus and spam corpus, FILTER the biagrams that shows on both email.
    ham_corpus = {}
    spam_corpus = {}
    
    count = 0
    same = 0
    for key in ham.keys():
        if key in spam.keys():
            count += 1
            ham_corpus[key] = ham[key]
            spam_corpus[key] = spam[key]
            if ham[key] == spam[key]:
                same += 1
    print ("There are {} of biagrams in this dataset. {} of them has the same frequency in spam and ham text".format(count,same)) 
    # bigrams needs to shows up in both spam and ham text for label
    
    ham_df = pd.Series(ham_corpus)
    ham_df = ham_df.to_frame().reset_index()
    ham_df.columns = ["bigram", "ham_frequency"]
    spam_df = pd.Series(spam_corpus)
    spam_df = spam_df.to_frame().reset_index()
    spam_df.columns = ["bigram", "spam_frequency"]
    # raw data, spam/ham corpus
    corpus = pd.merge(ham_df,spam_df)

    # possibility calculation for each bigram
    """
    P (label | bigram_pair) = P(bigram_pair|label) * P(label) / P(bigram_pair)
    bigram_pair  P(bigram_pair|spam)  P(spam)   P(bigram_pair)  

    * P(bigram_pair|spam) = Freq_spam_corpus(column_value) / num_of_row_spam
    * P(spam) = num_of_row_spam / (num_of_row_spam + num_of_row_Notspam)
    * P(bigram_pair) = (Freq_spam_corpus + Freq_Notspam_corpus) / (num_of_row_spam + num_of_row_Notspam)
    """
    bayes_corpus = corpus.copy()
    bayes_corpus['p_bigram_spam'] = bayes_corpus['spam_frequency'].apply(lambda x: x/n_spam)
    p_spam = n_spam / (n_spam + n_ham) # p_spam = 0.1340150699677072
    print ("There are {} spam message, {} non spam.".format(n_spam,n_ham))
    bayes_corpus['bigram_count'] = bayes_corpus['spam_frequency'] + bayes_corpus['ham_frequency']
    bayes_corpus['p_bigram'] = bayes_corpus['bigram_count'] / (n_spam+n_ham)

    # main possibility P(spam | bigram)
    bayes_corpus['p_spam_bigram'] = bayes_corpus['p_bigram_spam'] * p_spam / bayes_corpus['p_bigram']
    #print (bayes_corpus)

    return bayes_corpus

def main():
    alist = [line.rstrip() for line in open("source/collection", "r",encoding="ISO-8859-1")]
    print ("Size of dataset: ",len(alist))
    predict_size = 1000
    main_df = generate_corpus(alist)
    # generate probability dictionary for each bigrams based on panda datafram calculation
    df = main_df.copy()
    bigram_list = df["bigram"].tolist()
    p_spam_bigram = df["p_spam_bigram"].tolist()
    bigram_dict = dict(zip(bigram_list,p_spam_bigram))
    #print (bigram_dict)
    
    # prediction
    """
    # randomly select 1000 instences(raw) from source data, keep labels, generate bigrams for each sentence, 
    # lookup p_spam_bigram for each bigrams, multiply their p, product of p > 0.5 => spam, else not spam.
    if result == label, correct += 1, else loss += 1
    """
    test = np.random.choice(len(alist),predict_size, replace=False)
    
    label_list = []
    prob_list = []
    for case in test:
        line = alist[case].split()
        label_list.append(line[0]) #label
        bigram = generate_bigrams(line[1:])  #text
        prob = []
        for i in bigram:
            if i in bigram_dict:
                prob.append(bigram_dict[i])
        if len(prob) != 0:
            ave = sum(prob)/len(prob)
        else:
            ave = 0
        prob_list.append(ave)
        # Note: generaly speaking, for each bigram pair in text, should find the multiple of them to calculate the probability.
        # However, since dataset is not large enough to refine the p_spam_bigram, this probability can be very small.
        # this makes the final probability can be 10^(-27). Therefore, use average of them instead.
    
    print ("range of prediction score: ({},{})".format(min(prob_list),max(prob_list)))
    """
    translate score for each message into span/ham
    if score > 0.5: spam, else: ham
    """
    # predict based on score
    predict = []
    for n in prob_list:
        if n > 0.5:
            predict.append("spam")
        else:
            predict.append("ham")
    correct = 0
    for i in range (len(predict)):
        if label_list[i] == predict[i]:
            correct += 1
    accuracy = correct / predict_size
    print ("Test Size: {}; Predict accuracy: {}".format(predict_size,accuracy))
    #newdict = dict(sorted(bigram_dict.items(), key=bigram_dict.keys(), reverse=True)[:5])
    print ("The top 5 frequent spam bigrams are: ")
    for w in sorted (bigram_dict, key=bigram_dict.get,reverse=True)[:5]:
        print (w, bigram_dict[w])
            


if "__name__" == main():
    main()
