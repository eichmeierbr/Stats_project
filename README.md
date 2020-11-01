

pip install --upgrade pyDOE


Installing Stable Baseline:

sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

cd stable-baselines

pip install -e .  

sudo apt-get install libtcmalloc-minimal4

export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
