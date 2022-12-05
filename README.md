# CS 520 better-smarter-faster

## Setup Guide for Anaconda

Following steps need to be performed to replicate the environment I am using on Miniconda:

Run the following on CLI (with miniconda installed) in the repo folder: conda env create -f environment.yml

Activate the created environment using: conda activate /Path/To/Your/env

## Additional non-python packages

I have also installed the following packages:

graphviz (used by pygraphviz - which in turn was used by networkx.nx_agraph.graphviz_layout)
