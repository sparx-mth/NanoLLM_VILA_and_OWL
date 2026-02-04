 # VILA 1.5-3B Live WebRTC Streaming on Jetson

This guide describes how to run **VILA 1.5-3B** with **NanoLLM** on a Jetson device, stream live camera input via **WebRTC**, and view it from a remote browser.

---

## Prerequisites

* NVIDIA Jetson device (AGX / Orin / Xavier)
* Docker installed
* `jetson-containers` installed
* USB camera connected (e.g. `/dev/video0`)
* Chrome or Chromium browser on the client machine

---

## 1. Connect to the Jetson

From your local machine:

```bash
ssh -X user@172.16.17.9
```

> Note: X11 forwarding is **not required** for WebRTC streaming, but SSH is used to launch the container.

---

## 2. Run the NanoLLM Docker Container

```bash
jetson-containers run -it \
  --publish 8080:8080 \
  --volume /home/user/jetson-containers/data:/home/user/jetson-containers/data \
  --device=/dev/video0 \
  --device=/dev/video1 \
  nano_llm_custom /bin/bash
```

This exposes:

* Web UI on port **8050** (inside the container)
* WebRTC stream on port **8554**

---

## 3. Launch VILA 1.5-3B with Live Video Streaming

Inside the container:

```bash
python3 -m nano_llm.agents.video_query --api=mlc \
  --model Efficient-Large-Model/VILA1.5-3b \
  --max-context-len 256 \
  --max-new-tokens 32 \
  --video-input /dev/video0 \
  --video-output webrtc://@:8554/output
```

This:

* Captures live video from the camera
* Runs vision-language inference
* Streams video via WebRTC

*the video_query file is located in opt/NanoLLM/nano_llm/agents/video_query.py*
---

## 4. Open the Web Interface

From your **local machine browser**:

```
https://<JETSON_IP>:8050
```

Example:

```
https://172.16.17.9:8050
```

### Browser Requirements

* **Chrome or Chromium is strongly recommended**
* Disable WebRTC mDNS local IP masking:

```
chrome://flags/#enable-webrtc-hide-local-ips-with-mdns
```

Set it to **Disabled**, then restart the browser.

---

## 5. Required GStreamer Dependencies (If WebRTC Video Does Not Appear)

If the WebRTC connection is established but video does not render, install missing plugins **inside the container**:

```bash
apt-get update
apt-get install -y \
  libnice10 \
  gstreamer1.0-nice \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav
```

Then clear GStreamer caches:

```bash
rm -rf /root/.cache/gstreamer-1.0
rm -rf ~/.cache/gstreamer-1.0
```

And ensure headless operation:

```bash
unset DISPLAY
```

Restart the NanoLLM command after this.

---

## Notes & Troubleshooting

* The system is designed to run **headless** (no local display required)
* WebRTC video playback depends on browser autoplay policies
* Chrome requires video to be **muted + autoplay** (handled by the NanoLLM UI)
* If the page loads but the video stays black, refresh once after startup

---

## Summary

This setup enables:

* Live camera capture on Jetson
* Vision-language inference with VILA 1.5-3B
* Real-time WebRTC streaming to a remote browser

The pipeline is stable once the correct GStreamer plugins and browser flags are configured.

---

Happy hacking 🚀
