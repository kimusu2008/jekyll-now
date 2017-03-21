---
layout: post
title: Clustering of word2vec word vectors
---

I used the word2vec tool and a small Bahasa text corpus (3MB) as input and produced the word vectors as output. It first constructs a vocabulary from the training text data and then learns vector representation of words. The resulting word vector file can be used as features in many natural language processing and machine learning applications.

In this simple experiment, the word vectors have the size of ~6800 rows (vocab size) and 100 rows. Randomly sampled 500 rows (words) and used R's clusplot function (cluster package) for plotting the 2-dimensions cluster plot (the clustplot function has the dimensions reduction process built in and it uses the first two components for plotting). The plot looks like:

![_config.yml]({{ site.baseurl }}/images/pic3.png)

It is hard to comprehend the grouping results hence I put them into lists as shown below where each circle denotes a grouping which encloses syntactically or semantically words.

![_config.yml]({{ site.baseurl }}/images/pic1.png)

![_config.yml]({{ site.baseurl }}/images/pic2.png)


Some groupings seem logical, for instance the group {"inspektor" "9" "15" "januari" "februari" "79" "17" "12" "19"}. Once can categorise this is a group consists of mostly numeric numbers or month numbers. The other example is {berpengetahuan" "pergigian" "pembelajaran" "graduan" "saintifik"} in which the words are related to education. Bear in mind the model was trained using a small text corpus, so I anticipate much improved results given a larger corpus.

Alternatively, I tried the t-SNE dimension reduction package, the sample codes are shown below:

![_config.yml]({{ site.baseurl }}/images/pic4-1.png)

and clustering results:

![_config.yml]({{ site.baseurl }}/images/pic5.png)

Bear in mind that clustering quality and pca (parameter-free method) vs t-SNE (parameter) are not the points of discussion in this exercise.

