#!/bin/bash

base=ott2.8
mkdir $base
cd $base
wget http://files.opentreeoflife.org/ott/$base.tgz
tar xvfz $base.tgz
rm $base.tgz
