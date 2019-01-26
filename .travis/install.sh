#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    { brew install --upgrade gcc || true; }
    case "${TOXENV}" in
	py27)
	    { brew install --upgrade python@2 || true; }
	    ;;
	py37)
	    { brew install --upgrade python || true; }
	    ;;
    esac
    
    CC=$(which gcc-8) pip install .
else
    pip install Cython
    pip install nose
    pip install .
fi
