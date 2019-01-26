#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]
then
    { brew install --upgrade gcc || true; }
    CC=$(which gcc-8)
    case "${TOXENV}" in
	py27)
	    /usr/local/bin/pip2 install .
	    ;;
	py37)
	    /usr/local/bin/pip3 install .
	    ;;
    esac
    
else
    pip install Cython
    pip install nose
    pip install .
fi
