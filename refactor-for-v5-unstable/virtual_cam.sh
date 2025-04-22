sudo modprobe v4l2loopback video_nr=10 card_label="VirtualCam" exclusive_caps=1
# verify:
ls -l /dev/video10
