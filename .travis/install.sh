#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    { brew install --upgrade gcc || true; }
    case "${TOXENV}" in
	py27)
	    alias pip=/usr/local/bin/pip2
	    ;;
	py37)
	    alias pip=/usr/local/bin/pip3
	    ;;
    esac
    pip install --upgrade numpy
    pip install --upgrade scipy
    pip install Cython
    pip install nose
    CC=$(which gcc-8) pip install .
else
    pip install Cython
    pip install nose
    pip install .
fi

