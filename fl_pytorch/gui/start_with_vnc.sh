#!/usr/bin/env bash
# export QT_QPA_PLATFORM="vnc"

# First argument - port
# Second argument - screen size

listen_port=${1:-5209}
screen_size=$2{:-1920x1080}

python "start.py" -platform "vnc:size=${screen_size}:port=${listen_port}"

