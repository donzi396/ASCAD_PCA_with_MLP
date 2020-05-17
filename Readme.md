## <a name=ASCAD_PCA_with_MLP"> ASCAD_PCA_with_MLP

## Copyright and license

Copyright (C) 2020, Daniel Direktor (<mailto:direktor@campus.technion.ac.il>)
The companion python script is placed into the public domain.

## About

This script is in continuance to the article ["Study of Deep Learning Techniques for Side-Channel Analysis and Introduction to ASCAD Database"](https://eprint.iacr.org/2018/053.pdf) (available on the [eprints](https://eprint.iacr.org)) and its complementary repository [ASCAD](https://github.com/ANSSI-FR/ASCAD).

Reading the article and downloading the ASCAD scripts is essential for running this script.

## The tested model

As shown in the article, the use of CNN is effective even with desynchronized signals but has a relatively high run-time.
Therefore I tried to examine the use of a pre-processing stage with PCA before running a MLP.
Using this model, so I hoped, will achieve preferred performance with a CNN near accuracy.
I will mention that the article explicitly rejects the usage of such a model (section 3.3.3) but the article disregards , as far as I can tell, the performance and noise-filtering options coming out of the model.

## The script

The ASCAD_PCA_with_MLP.py script is designated to check the efficiency of the described model for side channel attacks.
The script runs the model many times with each possible combination with these parameters:
 *  PCA components (10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700)
 *  Desynchronization levels (0,50,100)
 *  Number of epochs (100,200,300,400,500)
The script output is graphs (accuracy to number of traces) as PNG image files.