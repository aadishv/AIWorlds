#!/bin/bash

BEANPOPPER="BeanPopper"
JETSONAP="jetsonAP"
# How long to wait (seconds) before nmcli gives up on BeanPopper
WAIT_TIME=5

# 1) Try to activate BeanPopper (uses saved creds)
if nmcli -w $WAIT_TIME connection up "$BEANPOPPER"; then
    echo "✅ Connected to $BEANPOPPER"
    # tear down AP if it’s up
    nmcli connection down "$JETSONAP" 2>/dev/null
else
    echo "⚠️  Could not connect to $BEANPOPPER—starting AP mode"
    # tear down any client link, then bring up AP
    nmcli connection down "$BEANPOPPER" 2>/dev/null
    nmcli connection up   "$JETSONAP"
fi
