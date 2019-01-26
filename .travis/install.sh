#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    brew install gcc
    case "${TOXENV}" in
	py27)
	    brew install python@2
	    ;;
	py37)
	    brew install python
	    ;;
    esac
    
    CC=$(which gcc) pip install .
else
    pip install Cython
    pip install nose
    pip install .
fi
