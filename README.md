# classification-using-random-forest-a-comparative-study

## 1. Overview
Classification is a supervised learning approach used for identifying unseen objects
based on heuristic data. Several models are available and random forest (RF) is one
of those. In early days of its establishment only majority votes were counted from
the trees in a forest to predict a class label. Later some works had done assigning
weights to the trees for better accuracy. In those tasks the result was compared with
original paper of random forest and other algorithms like - CART, ID3, Naive Bayes
etc. In this work, comparison among three different approaches of weighted random
forests along with the original algorithm is performed. Analysis shows that different
algorithm works better in different types of data-sets.

## 2. Dataset
10 datasets are used for this work, collected from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/index.php). Datasets are rearranged in comma(,) separated text format; i.e.- class label, predictor 1, predictor 2, . . ., predictor n. So, applying any new dataset should be represented in this format to run.

## 3. How To Use
Prerequisites : Git, Java(TM) SE 1.8 Runtime Environment, Ecplise Luna.

Steps to follow-
  - Create a new Java Project in Eclipse.
  - Clone the repository or download as zip (in this case the zip file needed to be unzipped and imported in the project folder).
  - Open the 'Main' file of desired package and run the java code.

## 4. Output
For successful run, accuracy rate will be shown in the console along with correct and incorrect class label counts.

## 5. Links
Here are the links of implemented papers-
1. [Random Forests](https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf)
2. [A Weighted Random Forests Approach to Improve Predictive Performance](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3912194/)
3. [Using Random Forest to Learn Imbalanced Data](http://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)
4. [Trees Weighting Random Forest Method for Classifying High-Dimensional Noisy Data](http://ieeexplore.ieee.org/document/5704290/)
