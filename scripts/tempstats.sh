[ -r "/sys/class/thermal/thermal_zone0/temp" ] && awk '{printf "cpu: %.2f °C\n", $1/1000}' /sys/class/thermal/thermal_zone0/temp || echo "Error: Cannot read temperature from /sys/class/thermal/thermal_zone0/temp. Check path and permissions."
# same for gpu
[ -r "/sys/class/thermal/thermal_zone1/temp" ] && awk '{printf "gpu: %.2f °C\n", $1/1000}' /sys/class/thermal/thermal_zone1/temp || echo "Error: Cannot read temperature from /sys/class/thermal/thermal_zone1/temp. Check path and permissions."
