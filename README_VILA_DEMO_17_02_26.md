  ssh -X user@172.16.17.9
   jetson-containers run -it   --publish 8080:8080   --volume /home/user/jetson-containers/data:/home/user/jetson-containers/data  --device=/dev/video0 --device=/dev/video1 nano_llm_custom /bin/bash
  python3 -m nano_llm.agents.video_query --api=mlc \
    --model Efficient-Large-Model/VILA1.5-3b \
    --max-context-len 256 \
    --max-new-tokens 32 \
    --video-input /dev/video0 \
    --video-output webrtc://@:8554/output


Then navigate your browser to https://<IP_ADDRESS>:8050 after launching it with your camera. Using Chrome or Chromium is recommended for a stable WebRTC connection, with chrome://flags#enable-webrtc-hide-local-ips-with-mdns disabled.

if needed:
inside the container:
 apt-get update
apt-get install -y   libnice10   gstreamer1.0-nice   gstreamer1.0-plugins-bad   gstreamer1.0-plugins-good   gstreamer1.0-plugins-ugly   gstreamer1.0-libav
rm -rf /root/.cache/gstreamer-1.0
rm -rf ~/.cache/gstreamer-1.0
unset DISPLAY



