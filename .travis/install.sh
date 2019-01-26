#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    brew install gcc
    case "${TOXENV}" in
	py27)
	    { brew install upgrade python@2 || true }
	    ;;
	py37)
	    { brew install upgrade python || true }
	    ;;
    esac
    
    CC=$(which gcc) pip install .
else
    pip install Cython
    pip install nose
    pip install .
fi
