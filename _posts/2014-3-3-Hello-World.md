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
![_config.yml]({{ site.baseurl }}/images/config.png)

The easiest way to make your first post is to edit this one. Go into /_posts/ and update the Hello World markdown file. For more instructions head over to the [Jekyll Now repository](https://github.com/barryclark/jekyll-now) on GitHub.

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"
