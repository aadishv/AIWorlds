echo $(sed -n '2 p' config.txt | tr -d '\n') | sudo -S echo "" > /dev/null 2>/dev/null

sudo systemctl stop vexai 2> /dev/null
# gather python3.6 PIDs (if any)
py_pids=$(pgrep -x python3.6 || true)
if [[ -n "$py_pids" ]]; then
    if [[ "$1" == "kill" ]]; then
        echo $py_pids | xargs -r sudo kill -9
    else
        echo $py_pids | xargs -r sudo kill -2
    fi
fi
