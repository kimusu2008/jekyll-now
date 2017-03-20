---
layout: post
title: Trained a word2vec (word embeddings model) model for Bahasa and Singlish !
---

In this post I’m going to show how the word2vec (word embeddings model) encapsulates and delivers the contextual information without using any prior knowledge. I first encountered word2vec back in 2013 and had written a white paper for finding word similarities using a [different type of neural network](http://www.neuramatix.com/NeuraBASE%20for%20Finding%20Word%20Similarities.pdf).

First, I get a pretrained Word2Vec model up and running in Python to play with. The python package called gensim is used for this experiment, it is a popular NLP package with good documentation and tutorials. Someone claims that it the word2vec package of gensim is 7x faster than a word2vec program built using numpy (see bottom for the word2vec script), I used a pretrained model which consists of word vectors for a vocabulary of 3 million words and phrases (trained on approximately 100 billion words from a Google News dataset). After loaded the model, I asked the following questions, it returns both semantically and syntactically similar words. The answers are ranked according to the consine distance (between 0 and 1, with 1 denotes the most similar)

**Q1. What are the most similar words to "malaysia" - 
Ans: [(singapore', 0.6477215886116028), (australia', 0.6320846080780029), (uk', 0.6232705116271973), (canada', 0.621403694152832), (os', 0.6194831132888794), (south_africa', 0.6166156530380249), (ireland', 0.6131253242492676), (usa', 0.6127300262451172), (usb', 0.6124882102012634), (india', 0.6090297698974609)]**

**Q2. What words are similar to "bank" - 
Ans: [(banks', 0.7440758943557739), (banking', 0.690161406993866), (Bank', 0.6698698401451111), (lender', 0.6342284679412842), (banker', 0.6092954277992249), (depositors', 0.6031532287597656), (mortgage_lender', 0.579797625541687), (depositor', 0.5716427564620972), (BofA', 0.5714625120162964), (Citibank', 0.5589520931243896)]**

**Q3. What are the most similar words to "fly" - 
Ans: [(flying', 0.6795352697372437), (flew', 0.627173900604248), (flies', 0.6233824491500854), (flys', 0.6012992858886719), (flown', 0.567908525466919), (Fly', 0.564663290977478), (flight', 0.5158863067626953), (Nashdribbledupcourt', 0.5138447880744934), (Hyliradryskin', 0.49056071043014526), (fl_y', 0.47577089071273804)]**

**The word2vec model is also good in solving similar analogies/tasks, for instance, 
Q1: The analogy of France to paris, what's the equivalent of England ? (country to capital) 
Ans: [(birmingham', 0.5499223470687866), (essex', 0.5299263000488281), (manchester', 0.5256893634796143)] **
** Expecting London, perhaps need bigger corpus to train this analogy**

**Q1: The analogy of bake to baking, what's the equivalent of walk ? (verb to present participle) 
Ans: [('Running', 0.522030234336853), ('ran', 0.49774789810180664), ('runing', 0.479981392621994)]**
** Expecting walking, perhaps need bigger corpus to train this analogy**

Next, as an experiment, I trained a word2vec model using a small and cleaned 3.1MB Malay text file and tested the model:

**Q1. What are the most similar words to "Menteri" - 
Ans: [(Perdana', 0.9615239500999451), (Ketua', 0.9384366273880005), (disandang', 0.9335119724273682), (Setiausaha', 0.9269030094146729), (Mahathir', 0.9268919229507446), (Timbalan', 0.9260348081588745), (Datuk', 0.9258796572685242), (Presiden', 0.9233565330505371), (DYMM', 0.9228272438049316), (Seri', 0.9224402904510498)]**

I trained a word2vec model using a small and cleaned 3.1MB Malay text file and tested the model:

**Q1. What are the most similar words to "Menteri" - 
Ans: [(Perdana', 0.9615239500999451), (Ketua', 0.9384366273880005), (disandang', 0.9335119724273682), (Setiausaha', 0.9269030094146729), (Mahathir', 0.9268919229507446), (Timbalan', 0.9260348081588745), (Datuk', 0.9258796572685242), (Presiden', 0.9233565330505371), (DYMM', 0.9228272438049316), (Seri', 0.9224402904510498)]**

I also tried the same with a very small and clnead 57KB Singlish text which I had crawled from http://singlishbible.wikia.com/ :

**Q1. What are the most similar words to "simi" - 
Ans: [(one', 0.9996679425239563), (to', 0.9996678829193115), (you', 0.999666154384613), (God', 0.9996645450592041), (The', 0.9996629953384399), (and', 0.999659538269043), (all', 0.9996594190597534), (I', 0.9996591210365295), (u'will', 0.9996554255485535), (know', 0.9996553659439087)]**

**Q2. What are the most similar words to "liddat" - 
Ans: [(say', 0.9998219013214111), (He', 0.9998188614845276), (God', 0.9998164772987366), (got', 0.9998048543930054), (that', 0.9998034834861755), (and', 0.9998010993003845), (Then', 0.9998006820678711), (also', 0.9997992515563965), (they', 0.9997990131378174), (in', 0.9997986555099487)]**

**Q3. What are the most similar words to "kena" - 
Ans: [(the', 0.9999566078186035), (and', 0.9999526739120483), (one', 0.9999485015869141), (they', 0.9999473094940186), (of', 0.9999468326568604), (in', 0.9999450445175171), (got', 0.9999409914016724), (God', 0.9999405145645142), (Then', 0.9999399185180664), (say', 0.9999397993087769)]**

**Q4. What are the most similar words to "towkay" - 
Ans: [(the', 0.9998369216918945), (one', 0.9998359680175781), (to', 0.9998290538787842), (his', 0.9998285174369812), (Jesus', 0.9998282194137573), (he', 0.9998277425765991), (God', 0.9998258352279663), (and', 0.9998248815536499), (you', 0.9998229742050171), (in', 0.9998221397399902)]**

** Irrelevant results as expected given a small text file most of the results returned are of determiner or conjunction type

More to come in the upcoming posts....



`import json 
import numpy as np 
import theano 
import theano.tensor as T 
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle 
from datetime import datetime 
from util import findanalogies as _findanalogies

import os 
import sys 
sys.path.append(os.path.abspath('..')) 
from rnnclass.util import getwikipedia_data

def sigmoid(x): 
return 1 / (1 + np.exp(-x))

def init_weights(shape): 
return np.random.randn(*shape).astype(np.float32) / np.sqrt(sum(shape))

class Model(object): 
def init(self, D, V, contextsz): self.D = D # embedding dimension self.V = V # vocab size self.contextsz = context_sz

def _get_pnw(self, X):
    calculate Pn(w) - probability distribution for negative sampling
    basically just the word probability ^ 3/4
    word_freq = {}
    word_count = sum(len(x) for x in X)
    for x in X:
        for xj in x:
            if xj not in word_freq:
                word_freq[xj] = 0
            word_freq[xj] += 1
    self.Pnw = np.zeros(self.V)
    for j in xrange(2, self.V): # 0 and 1 are the start and end tokens, we won't use those here
        self.Pnw[j] = (word_freq[j] / float(word_count))**0.75
    # print "self.Pnw[2000]:", self.Pnw[2000]
    assert(np.all(self.Pnw[2:] > 0))
    return self.Pnw

def _get_negative_samples(self, context, num_neg_samples):
    # temporarily save context values because we don't want to negative sample these
    saved = {}
    for context_idx in context:
        saved[context_idx] = self.Pnw[context_idx]
        # print "saving -- context id:", context_idx, "value:", self.Pnw[context_idx]
        self.Pnw[context_idx] = 0
    neg_samples = np.random.choice(
        xrange(self.V),
        size=num_neg_samples, # this is arbitrary - number of negative samples to take
        replace=False,
        p=self.Pnw / np.sum(self.Pnw),
    )
    # print "saved:", saved
    for j, pnwj in saved.iteritems():
        self.Pnw[j] = pnwj
    assert(np.all(self.Pnw[2:] > 0))
    return neg_samples

def fit(self, X, num_neg_samples=10, learning_rate=10e-5, mu=0.99, reg=0.1, epochs=10):
    N = len(X)
    V = self.V
    D = self.D
    self._get_pnw(X)

    # initialize weights and momentum changes
    self.W1 = init_weights((V, D))
    self.W2 = init_weights((D, V))
    dW1 = np.zeros(self.W1.shape)
    dW2 = np.zeros(self.W2.shape)

    costs = []
    cost_per_epoch = []
    sample_indices = range(N)
    for i in xrange(epochs):
        t0 = datetime.now()
        sample_indices = shuffle(sample_indices)
        cost_per_epoch_i = []
        for it in xrange(N):
            j = sample_indices[it]
            x = X[j] # one sentence

            # too short to do 1 iteration, skip
            if len(x) < 2 * self.context_sz + 1:
                continue

            cj = []
            n = len(x)
            for jj in xrange(n):

                # do the updates manually
                Z = self.W1[x[jj],:] # note: paper uses linear activation function

                start = max(0, jj - self.context_sz)
                end = min(n, jj + 1 + self.context_sz)
                context = np.concatenate([x[start:jj], x[(jj+1):end]])
                # NOTE: context can contain DUPLICATES!
                # e.g. "<UNKOWN> <UNKOWN> cats and dogs"
                context = np.array(list(set(context)), dtype=np.int32)
                # print "context:", context

                posA = Z.dot(self.W2[:,context])
                pos_pY = sigmoid(posA)

                neg_samples = self._get_negative_samples(context, num_neg_samples)

                # technically can remove this line now but leave for sanity checking
                # neg_samples = np.setdiff1d(neg_samples, Y[j])
                # print "number of negative samples:", len(neg_samples)
                negA = Z.dot(self.W2[:,neg_samples])
                neg_pY = sigmoid(-negA)
                c = -np.log(pos_pY).sum() - np.log(neg_pY).sum()
                cj.append(c / (num_neg_samples + len(context)))

                # positive samples
                pos_err = pos_pY - 1
                dW2[:, context] = mu*dW2[:, context] - learning_rate*(np.outer(Z, pos_err) + reg*self.W2[:, context])

                # negative samples
                neg_err = 1 - neg_pY
                dW2[:, neg_samples] = mu*dW2[:, neg_samples] - learning_rate*(np.outer(Z, neg_err) + reg*self.W2[:, neg_samples])

                self.W2[:, context] += dW2[:, context]
                # self.W2[:, context] /= np.linalg.norm(self.W2[:, context], axis=1, keepdims=True)
                self.W2[:, neg_samples] += dW2[:, neg_samples]
                # self.W2[:, neg_samples] /= np.linalg.norm(self.W2[:, neg_samples], axis=1, keepdims=True)

                # input weights
                gradW1 = pos_err.dot(self.W2[:, context].T) + neg_err.dot(self.W2[:, neg_samples].T)
                dW1[x[jj], :] = mu*dW1[x[jj], :] - learning_rate*(gradW1 + reg*self.W1[x[jj], :])

                self.W1[x[jj], :] += dW1[x[jj], :]
                # self.W1[x[jj], :] /= np.linalg.norm(self.W1[x[jj], :])

            cj = np.mean(cj)
            cost_per_epoch_i.append(cj)
            costs.append(cj)
            if it % 100 == 0:
                sys.stdout.write("epoch: %d j: %d/ %d cost: %f\r" % (i, it, N, cj))
                sys.stdout.flush()

        epoch_cost = np.mean(cost_per_epoch_i)
        cost_per_epoch.append(epoch_cost)
        print "time to complete epoch %d:" % i, (datetime.now() - t0), "cost:", epoch_cost
    plt.plot(costs)
    plt.title("Numpy costs")
    plt.show()

    plt.plot(cost_per_epoch)
    plt.title("Numpy cost at each epoch")
    plt.show()

def fitt(self, X, num_neg_samples=10, learning_rate=10e-5, mu=0.99, reg=0.1, epochs=10):
    N = len(X)
    V = self.V
    D = self.D
    self._get_pnw(X)

    # initialize weights and momentum changes
    W1 = init_weights((V, D))
    W2 = init_weights((D, V))
    W1 = theano.shared(W1)
    W2 = theano.shared(W2)

    thInput = T.iscalar('input_word')
    thContext = T.ivector('context')
    thNegSamples = T.ivector('negative_samples')

    W1_subset = W1[thInput]
    W2_psubset = W2[:, thContext]
    W2_nsubset = W2[:, thNegSamples]
    p_activation = W1_subset.dot(W2_psubset)
    pos_pY = T.nnet.sigmoid(p_activation)
    n_activation = W1_subset.dot(W2_nsubset)
    neg_pY = T.nnet.sigmoid(-n_activation)
    cost = -T.log(pos_pY).sum() - T.log(neg_pY).sum()

    W1_grad = T.grad(cost, W1_subset)
    W2_pgrad = T.grad(cost, W2_psubset)
    W2_ngrad = T.grad(cost, W2_nsubset)

    W1_update = T.inc_subtensor(W1_subset, -learning_rate*W1_grad)
    W2_update = T.inc_subtensor(
        T.inc_subtensor(W2_psubset, -learning_rate*W2_pgrad)[:,thNegSamples], -learning_rate*W2_ngrad)
    # 2 updates for 1 variable
    # http://stackoverflow.com/questions/15917849/how-can-i-assign-update-subset-of-tensor-shared-variable-in-theano
    # http://deeplearning.net/software/theano/tutorial/faq_tutorial.html
    # https://groups.google.com/forum/#!topic/theano-users/hdwaFyrNvHQ

    updates = [(W1, W1_update), (W2, W2_update)]

    train_op = theano.function(
        inputs=[thInput, thContext, thNegSamples],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True,
    )

    costs = []
    cost_per_epoch = []
    sample_indices = range(N)
    for i in xrange(epochs):
        t0 = datetime.now()
        sample_indices = shuffle(sample_indices)
        cost_per_epoch_i = []
        for it in xrange(N):
            j = sample_indices[it]
            x = X[j] # one sentence

            # too short to do 1 iteration, skip
            if len(x) < 2 * self.context_sz + 1:
                continue

            cj = []
            n = len(x)
            for jj in xrange(n):

                start = max(0, jj - self.context_sz)
                end = min(n, jj + 1 + self.context_sz)
                context = np.concatenate([x[start:jj], x[(jj+1):end]])
                # NOTE: context can contain DUPLICATES!
                # e.g. "<UNKOWN> <UNKOWN> cats and dogs"
                context = np.array(list(set(context)), dtype=np.int32)
                neg_samples = self._get_negative_samples(context, num_neg_samples)

                c = train_op(x[jj], context, neg_samples)
                cj.append(c / (num_neg_samples + len(context)))

            cj = np.mean(cj)
            cost_per_epoch_i.append(cj)
            costs.append(cj)
            if it % 100 == 0:
                sys.stdout.write("epoch: %d j: %d/ %d cost: %f\r" % (i, it, N, cj))
                sys.stdout.flush()

        epoch_cost = np.mean(cost_per_epoch_i)
        cost_per_epoch.append(epoch_cost)
        print "time to complete epoch %d:" % i, (datetime.now() - t0), "cost:", epoch_cost

    self.W1 = W1.get_value()
    self.W2 = W2.get_value()

    plt.plot(costs)
    plt.title("Theano costs")
    plt.show()

    plt.plot(cost_per_epoch)
    plt.title("Theano cost at each epoch")
    plt.show()

def save(self, fn):
    arrays = [self.W1, self.W2]
    np.savez(fn, *arrays)
def main(): 
sentences, word2idx = getwikipediadata(nfiles=1, nvocab=2000) with open('w2v_word2idx.json', 'w') as f: json.dump(word2idx, f)

V = len(word2idx)
model = Model(80, V, 10)
model.fitt(sentences, learning_rate=10e-4, mu=0, epochs=5)
model.save('w2v_model.npz')
def findanalogies(w1, w2, w3, concat=True, wefile='w2vmodel.npz', w2ifile='w2vword2idx.json'): 
npz = np.load(wefile) W1 = npz['arr0'] W2 = npz['arr1']

with open(w2i_file) as f:
    word2idx = json.load(f)

V = len(word2idx)

if concat:
    We = np.hstack([W1, W2.T])
    print "We.shape:", We.shape
    assert(V == We.shape[0])
else:
    We = (W1 + W2.T) / 2

_find_analogies(w1, w2, w3, We, word2idx)
if name == 'main': 
main() for concat in (True, False): print "** concat:", concat findanalogies('king', 'man', 'woman', concat=concat) findanalogies('france', 'paris', 'london', concat=concat) findanalogies('france', 'paris', 'rome', concat=concat) findanalogies('paris', 'france', 'italy', concat=concat)`


![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"
