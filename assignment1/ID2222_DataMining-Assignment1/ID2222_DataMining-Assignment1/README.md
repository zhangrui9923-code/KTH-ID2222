# ID2222_DataMining

## Task

You are to implement the stages of finding textually similar documents based on Jaccard similarity using the shingling, minhashing, and locality-sensitive hashing (LSH) techniques and corresponding algorithms. The implementation can be done using any big data processing framework, such as Apache Spark or Apache Flink, or no framework, e.g., in Java, Python, etc. To test and evaluate your implementation, write a program that uses your implementation to find similar documents in a corpus of 5-10 or more documents, such as web pages or emails.

The stages should be implemented as a collection of classes, modules, functions, or procedures, depending on the framework and the language of your choice. Below, we describe sample classes implementing different stages of finding textually similar documents. You do not have to develop the exact same classes and data types described below. Feel free to use data structures that suit you best.

1. A class Shingling that constructs k–shingles of a given length k (e.g., 10) from a given document, computes a hash value for each unique shingle and represents the document in the form of an ordered set of its hashed k-shingles.
2. A class CompareSets computes the Jaccard similarity of two sets of integers – two sets of hashed shingles.
3. A class MinHashing that builds a minHash signature (in the form of a vector or a set) of a given length n from a given set of integers (a set of hashed shingles).
4. A class CompareSignatures estimates the similarity of two integer vectors – minhash signatures – as a fraction of components in which they agree.
5. (Optional task for an extra 2 bonus points) A class LSH that implements the LSH technique: given a collection of minhash signatures (integer vectors) and a similarity threshold t, the LSH class (using banding and hashing) finds candidate pairs of signatures agreeing on at least a fraction t of their components.

To test and evaluate your implementation's scalability (the execution time versus the size of the input dataset), write a program that uses your classes to find similar documents in a corpus of 5-10 documents. Choose a similarity threshold s (e.g., 0,8) that states that two documents are similar if the Jaccard similarity of their shingle sets is at least s.

## Dataset

<!-- [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

[Twenty Newsgroup](https://archive.ics.uci.edu/dataset/113/twenty+newsgroups) -->

The chapters from Yes, Prime Minister.

## How to run

```
uv sync
```

Then use jupyter notebook to run the code.
