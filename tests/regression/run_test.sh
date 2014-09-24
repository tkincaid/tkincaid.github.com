#!/bin/bash

set -e

CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd $CDIR/../..

export PYTHONPATH="`pwd`"

if [ $# -ne 2 ]
then
    echo "Requires two arguments"
    exit 1
fi

if [ "x$1" = "xtrue" ]
then
    if [ "x$2" = "xtrue" ]
    then
        skip_complete="--skip"
    else
        skip_complete=""
    fi
    python ./tests/regression/test_blueprints.py --fit $skip_complete
else
    python ./tests/regression/test_blueprints.py
fi
