# TODO — Orin NX rollout

State (2026-10-11): vLLM + NanoOWL + DA3 run together on the NX, start in order at boot
(`orin_nx/`), pipeline + display verified end to end with the drone camera.
Exports of the 3 images + model data (~32 GB, SHA256SUMS) are on the dev laptop (`~/jetson_export`).

1. **Copy the same configuration to Rooster R1 and R2**
   - Each board needs JetPack, Docker with the nvidia runtime, README section 0 (GUI off), the repo.
   - `docker load` the 3 images, extract `data.tar.gz` (README "Backups and cloning").
   - Per board: new IP, delete the copied DA3 `.engine` and rebuild with `trtexec` (README 5),
     that camera's `cam.yaml`, install the startup service (README 6).
   - Reach boards through the RPi jump host (README 8).
2. **Repeat the model + camera tests on each board**
   - `v4l2-ctl --list-devices` — capture node name may differ; capture -> pipeline -> display.
3. ~~**Service that deletes old frames**~~ — decided: count-based (keep last N run folders),
   not age-based, since this board's clock resets at boot (README section 7) and an
   age-based rule would delete fresh frames the moment the clock looks decades off.
   `orin_nx/cleanup_frames.sh` + `orin-nx-cleanup.service`/`.timer` (hourly). Still open:
   only prunes `comm_manager_vllm.py`'s output triplets (`<run>`/`<run>_ann`/`<run>_depth`);
   the raw `auto_capture_interval.py` capture root needs its own instance of the same script.
4. ~~**Smaller capture images**~~ — decided: 640x360 (now the `auto_capture_interval.py`
   default), matching `docker/da3/camera_info_640x360.yaml`'s existing calibration and
   confirmed available at a full 30fps in MJPG. Still open: whether full-size originals
   are needed later (not currently kept anywhere if so).
5. **Inspect depth values and the DA3 size**
   - Compare `DEPTH_ANYTHING` distances to known distances; is 252x448 enough?
   - Lens is a fisheye: calibrate (OpenCV fisheye) and undistort before DA3, then fix `cam.yaml`.
6. Smaller items
   - Fix the `/dev/HD_CAMERA` udev rule (currently -> `video1`, the metadata node; want `video0`).
   - Update README "Known issues" after the above; remove the stale `ollama` service if unused.
