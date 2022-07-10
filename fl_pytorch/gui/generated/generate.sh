#!/usr/bin/env bash

# Add to path place with pyuic5. In windows OS it's the same place where pip3 is storing
# set -o xtrace

# This part of script is dedicated for convert *.ui files into Python .py file via using the pyuic5 tool.
for f in `ls -1 ./../forms`; do
    echo "Generate code from ui file: $f"
    pyuic5 -x ./../forms/${f} -o `basename ${f%.ui}.py`
done

# This part of script is dedicated for convert resource Qt files,
# which are *.qrc files in a simple XML format into Python
for f in `ls -1 ./../resources/*.qrc`; do
    echo "Generate code from resource files: $f"
    pyrcc5 ./../resources/${f} -o `basename ${f%.qrc}_rc.py`
done
