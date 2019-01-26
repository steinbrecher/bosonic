#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    { brew install --upgrade gcc || true; }
    CC=$(which gcc-8)
    case "${TOXENV}" in
	py27)
	    alias pip=/usr/local/bin/pip2
	    ;;
	py37)
	    alias pip=/usr/local/bin/pip3
	    ;;
    esac
    
fi
pip install Cython
pip install nose
pip install .

