#!/bin/bash

mkdir ncbi
cd ncbi
wget ftp://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
tar xvfz taxdump.tar.gz
rm taxdump.tar.gz
