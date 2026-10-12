# TODO — Orin NX rollout

State (2026-10-11): vLLM + NanoOWL + DA3 run together on the NX, start in order at boot
(`orin_nx/`), pipeline + display verified end to end with the drone camera.
Exports of the 3 images + model data (~32 GB, SHA256SUMS) are on the dev laptop (`~/jetson_export`).

1. **Copy the same configuration to Rooster R1 and R2**
   - Each board needs JetPack, Docker with the nvidia runtime, README section 0 (GUI off).
   - `docker load` the 3 images, extract `data.tar.gz`, **and deploy this repo + its Python
     deps separately** (README "Backups and cloning" — the image/data export does not include
     the repo; found this the hard way bringing up R2, which had the containers but no code).
   - Per board: new IP, delete the copied DA3 `.engine` and rebuild with `trtexec` (README 5),
     that camera's `cam.yaml`, install the startup service (README 6), the drone-ID banner (README 9).
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
   - Resolution benchmarked on R2 (raw `trtexec`, isolated): `336x602` 224ms/4.46fps vs
     `364x644` 275ms/3.63fps. Chose `336x602` for the speed margin on a board that also runs
     vLLM+NanoOWL concurrently. Real HTTP-serving benchmark (20 requests, `/bbox_depth`,
     actual frame + NanoOWL detections) confirmed this holds up in practice: avg 259.5ms/req
     (~3.85fps), tight 254.5-263.5ms spread (no slow-tail spikes), only ~35ms overhead above
     the raw engine number. Not the pipeline's bottleneck either way - vLLM's own per-frame
     captioning (0.6-1.8s) is slower than DA3 at any of these sizes.
   - **R2 now runs a different DA3 resolution than documented (`336x602` vs the README's
     `252x448`)** - R1/R3 (or wherever `252x448` is still in use) are no longer on the
     "same configuration," which item 1 above assumes. Decide whether to standardize on one
     resolution across all boards before the R1/R3 rollout, or deliberately let each board's
     resolution vary (e.g. if their cameras/compute loads differ).
   - Compare `DEPTH_ANYTHING` distances to known distances - is either size accurate enough?
     (Saw some negative depth minimums in R2's debug log, e.g. `depth min=-0.632213` - likely
     boundary/background noise, not necessarily a problem, but worth checking during this.)
   - Lens is a fisheye: calibrate (OpenCV fisheye) and undistort before DA3, then fix `cam.yaml`.
6. Smaller items
   - Fix the `/dev/HD_CAMERA` udev rule (currently -> `video1`, the metadata node; want `video0`).
   - Update README "Known issues" after the above; remove the stale `ollama` service if unused.
7. **Clean up the repo** — a lot of what's at the repo root predates the Orin NX pipeline and
   isn't used by it. Candidates to review (confirm unused elsewhere before deleting, not just
   assumed):
   - Duplicate `comm_manager_vllm*`/`display_server*` variants (`comm_manager.py`,
     `comm_manager_vllm_batch.py`, `comm_manager_vllm_with_scene.py`,
     `display_server_depth.py`, `display_server_depth_and_one_image.py`,
     `display_server_pazzle.py`, `display_server_with_scene.py`) vs. the two actually in
     use (`comm_manager_vllm.py`, `display_server.py`).
   - `jetson_dashboard.py`, `jetson_tui.py`, `jetson_infra_tui.py` — AGX-era TUI tools.
   - `VILA/`, `VILA_live_streaming_on_jetson.md` — superseded by the vLLM/Qwen2-VL path.
   - `LLM/` — prompt-converter code for an unused text-only-LLM step (`comm_manager_vllm.py`'s
     own `VLLM_URL`/`VLLM_MODEL`/etc. globals are similarly vestigial from this same era).
   - `room_mapping/`, `mission_control/`, `apriltag/` — separate features, not part of this
     vision pipeline; confirm whether still used by another workflow before touching.
   - `comm_timings.log` — a log file, tracked in git; should be gitignored instead.
   - `demo_with_main*.drawio.png` — old diagrams, confirm still referenced anywhere first.
