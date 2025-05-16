#!/bin/bash
# no longer needed, used logrotate to keep filesize in check
# will remove previous log file.
#rm -f /home/vex/AIWorlds/comp3/output.txt

# debug
echo "[$(date)] cwd before cd: $(pwd)"    >> /home/vex/AIWorlds/comp3/output.txt
cd "/home/vex/AIWorlds/comp3"             >> /home/vex/AIWorlds/comp3/output.txt 2>&1
echo "[$(date)] cwd after cd:  $(pwd)"    >> /home/vex/AIWorlds/comp3/output.txt

#  since it will be run as service when jetson starts every time, no need for following
#  bash runner2.sh kill


# Set the required environment variables
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6


cd "/home/vex/AIWorlds/comp3"
/usr/bin/python3.6 main.py sim 
#/usr/bin/python3.6 main.py ser


# cd back-test
# python3.6 app.py
