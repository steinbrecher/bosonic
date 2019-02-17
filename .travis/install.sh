#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    { brew install gcc || true; }
    { brew link gcc || true; }
    { brew install libomp || true; }
    case "${TOXENV}" in
	py27)
	    alias pip=/usr/local/bin/pip2
	    ;;
	py37)
	    alias pip=/usr/local/bin/pip3
	    ;;
    esac
    pip install --upgrade --user numpy
    pip install --upgrade --user scipy
    pip install --user Cython
    pip install --user nose
    CC=gcc-8 pip install .
else
    pip install Cython
    pip install nose
    pip install .
fi

