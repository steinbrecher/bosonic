#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    CC=gcc-8 python setup.py test
else
    python setup.py test
fi
