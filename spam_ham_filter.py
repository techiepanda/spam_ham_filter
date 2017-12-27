#Naive Bayes algorithm to classify spam
#Used Multinomial Naive Bayes to detect spam/ham emails.
#Tried different smoothing values and got high accuracy with value 0.05
import sys
import math
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f1', help='training file in csv format', required=True)
    parser.add_argument('-f2', help='test file in csv format', required=True)
    parser.add_argument('-o', help='output labels for the test dataset', required=True)

    args = vars(parser.parse_args())

    training_file = args['f1']
    testing_file = args['f2']
    output_file = args['o']

    spam_dict = {}
    ham_dict = {}
    spamCount = 0
    hamCount = 0
    spamWordCount = 0
    hamWordCount = 0
    message = ""
    with open(training_file,"r") as f:
        emails = f.read().split("\n")
        for mail in emails:
            if len(mail) < 2:
                continue
            message = mail.split(" ")
            if len(message) > 2:
                if  message[1] == "ham" :
                    hamCount += 1
                    for i in range(2, len(message), 2):
                        try:
                            # Store the total count for each word appeared in ham emails.
                            ham_dict[message[i]] += float(message[i+1])
                        except KeyError:
                            ham_dict[message[i]] = float(message[i+1])
                        # Store the total count of words appeared in ham emails.
                        hamWordCount += float(message[i+1])
                else:
                    spamCount += 1
                    for i in range(2, len(message), 2):
                        try:
                            # Store the total count for each word appeared in spam emails.
                            spam_dict[message[i]] += float(message[i+1])
                        except KeyError:
                            spam_dict[message[i]] = float(message[i+1])
                        # Store the total count of words appeared in spam emails.
                        spamWordCount += float(message[i+1])
                
    vocabulary = set(spam_dict.keys()).union(set(ham_dict.keys()))
    vocabulary_size = len(vocabulary)  # Get the vocabulary size
    spam_dict.update((k, float(v) + 0.05/ spamWordCount + 0.05*vocabulary_size) for k,v in spam_dict.items())
    ham_dict.update((k, float(v) + 0.05/ hamWordCount + 0.05*vocabulary_size) for k,v in ham_dict.items())

    total_Count = spamCount + hamCount
    p_spam = float(spamCount) / total_Count
    p_ham = float(hamCount) / total_Count

    correctPred = 0
    incorrectPred = 0
    resMessage = ""
    smoothingParameter = 0.5
    with open(testing_file,"r") as f:
        emails = f.read().split("\n")
        for mail in emails:
            if len(mail) < 2:
                continue
            message = mail.split(" ")
            if len(message) > 2:
                resMessage += message[0]
                actualResult = message[1]
                ham_prob = 1.0
                spam_prob = 1.0
                for i in range(2, len(message), 2):
                    if len(message[i]) > 2:
                        # Adding  0.5 prob. for new words which are not seen during testing.
                        if message[i] in spam_dict:
                            spam_prob += float(message[i+1]) * math.log(spam_dict[message[i]])
                        else:
                            spam_prob += math.log(smoothingParameter)
                        if message[i] in ham_dict:
                            ham_prob += float(message[i+1]) * math.log(ham_dict[message[i]])
                        else:
                            ham_prob += math.log(smoothingParameter)
                spam_prob = spam_prob  + math.log(p_spam)
                ham_prob = ham_prob + math.log(p_ham)
                if ham_prob > spam_prob:
                    res = "ham"
                else:
                    res = "spam"
                if res == actualResult:
                    correctPred += 1
                else:
                    incorrectPred += 1
                resMessage += "," + res + "\n"

    with open(output_file,"w") as f:
        f.write(resMessage)

    print "Correct Labels: ", correctPred
    print "Incorrect Labels: ", incorrectPred
    print "Accuracy: ", (float(correctPred) * 100.0)/ (correctPred + incorrectPred), "%"
