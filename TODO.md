# TODO — Orin NX rollout

State (2026-10-11): vLLM + NanoOWL + DA3 run together on the NX, start in order at boot
(`orin_nx/`), pipeline + display verified end to end with the drone camera.
Exports of the 3 images + model data (~32 GB, SHA256SUMS) are on the dev laptop (`~/jetson_export`).

1. **Bring R1 up fresh and update R3** (R2 is now the working reference — see README
   "Cloning to R1/R3" for the full copy/run/don't-touch checklist)
   - R1 and R3 both need: JetPack, Docker with the nvidia runtime, README section 0 (GUI off).
   - Copy from R2: the 3 Docker images, `jetson-containers/data`, and DA3's `336x602` ONNX +
     `cam.yaml` (not the `.engine` — rebuild that per board). **And this repo itself + its
     Python deps** — found the hard way bringing up R2, which had the containers but no code,
     since the image/data export never included it.
   - Run per board (not copied): `pip install` the repo's deps, the DA3 `trtexec` rebuild
     (README 5), the drone-ID MOTD/prompt (README 9, identical commands everywhere), the
     startup service (README 6).
   - Do NOT copy: each board's own `~/iai_ws/src/docker/.env` (`ROS_DOMAIN_ID`/`DDS_IFACE`
     are already correct per board) or camera calibration if R1/R3 use a different physical
     camera unit than R2's.
   - Reach boards through the RPi jump host (README 8).
   - **R1 vLLM `--gpu-memory-utilization`: `0.63` for vLLM running alone** (not `0.52`) —
     confirmed on two independent clean reboots (522-533 MiB idle, `nvmap` page-pool at `0`,
     confirmed via `tegrastats`/the pagepool file *before* starting anything, `0.52` as the
     first and only thing to touch the GPU that boot): `0.52` failed both times with an almost
     identical deficit (`Available KV cache memory: -1.58` and `-1.6 GiB`), while `0.63`
     succeeded with `+1.93 GiB`.
     - **Important methodology finding, found while sanity-checking this against R2/R3:**
       `vllm_qwen` must be restarted with NanoOWL/DA3 **stopped** for its memory snapshot to
       be meaningful — restarting it while they're already loaded produces a false "needs
       more" result. R2 failed to even start at `0.63` (pre-flight check: only `7.92/15.29
       GiB` free) simply because `da3`/`now_eng`/`iai` were already running when `vllm_qwen`
       was restarted during this investigation. This explains a lot of apparent flakiness in
       earlier R2/R3 retests of `0.52` (mixed successes around `+0.2-0.4 GiB` and failures
       around `-1.6 GiB` — inconsistent restart conditions, not a stable number).
     - **This does not fully explain R1**, though: stopping NanoOWL/DA3 on R2 and retesting
       `0.52` isolated gave a solidly positive but thin result (`+0.24 GiB`) — consistent with
       R1 still needing more even under an equally isolated, apples-to-apples test. So R1
       genuinely appears to need more margin than R2, even once the confound above is
       controlled for. Root cause still not found despite ruling out (all identical to
       R2/R3): L4T version (`R36.4.4`), CMA size (256 MiB), GUI session, stray processes, GPU
       SM count/global memory (`trtexec`: `SMs: 8` / `15655 MiB` — not silicon binning), and
       kernel reserved-memory carveout (`sudo dmesg | grep -iE "reserved|carveout|memblock"`:
       R1 `514112K` reserved / `16521856K` total vs R2 `513664K` / R3 `513640K` — noise).
       Going further would mean instrumenting CUDA's own memory accounting directly (e.g. a
       raw `cudaMemGetInfo()` probe outside vLLM/Docker) — not worth it unless this resurfaces.
     - **Before trusting either number as final: get more isolated data points.** We only have
       *one* isolated R2 test (`+0.24 GiB`, thin) vs *two* isolated R1 tests (both clearly
       negative) — not enough of either to fully rule out run-to-run noise. Worth repeating
       the isolated (NanoOWL/DA3 stopped, clean reboot) test on R2/R3 a couple more times
       before concluding `0.52` is reliably safe there.
   - ~~**Full stack on R1**~~ — done: NanoOWL and DA3 (`336x602`, engine already built from an
     earlier session) brought up alongside vLLM (already running at `0.63`), following the
     documented order (vLLM already up, then NanoOWL, then DA3). All three answered `404`
     (healthy) afterward, and `tegrastats` settled at a stable `12353/15655 MB` (~3.2 GiB
     free headroom, not climbing) — `0.63` holds for the full stack on R1, not just vLLM
     alone. NanoOWL's baked-in CLIP/image-encoder TensorRT engine was built on a different
     board (image exported from R2) and warned `Using an engine plan file across different
     models of devices is not recommended` on startup — it ran fine regardless, but if
     NanoOWL detection quality/stability looks off on R1 later, rebuild that engine on R1
     specifically (same rule as DA3's engines).
   - ~~**R1 camera + end-to-end pipeline test**~~ — done. `v4l2-ctl --list-devices`: same
     layout as R2/R3 (`/dev/video0` capture, `/dev/video1` metadata). `captures/` needed
     `sudo chmod -R 777` once (root-owned from an earlier bind mount) before `iai` could
     write to it. `auto_capture_interval.py` at 640x360 confirmed streaming. Full pipeline
     (`comm_manager_vllm.py` -> NanoOWL -> `display_server.py`, viewed through an RPi SSH
     port-forward since R1 is only reachable via the jump host with the camera attached)
     confirmed working.
   - ~~**R1 startup service + cleanup timer**~~ — done. Both unit files are already generic
     (derive the captures path from `ROS_DOMAIN_ID`, no per-board edits needed), so install
     was identical to R2/R3. Verified with a real `sudo reboot`:
     `orin-nx-services` brought up vLLM -> NanoOWL -> DA3 in order automatically
     (`journalctl`: vLLM ~6 min, NanoOWL + DA3 ~1.5 min more, ~7.5 min total — matches R2's
     documented cold-boot timing), all three answering `404` afterward.
   - **R1 is now at full production parity with R2/R3**: full stack running, confirmed at
     `0.63`, camera + end-to-end pipeline verified, auto-boot verified. Still not done:
     DA3's `--depth-endpoint` wasn't included in the `comm_manager_vllm.py` test — depth
     hasn't been exercised end-to-end on R1 yet (same as the still-open item for R2/R3 in
     README section 5).
2. ~~**Repeat the model + camera tests on each board**~~ — R2/R3 done earlier; R1 done above.
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
   - ~~Standardize resolution across boards~~ — decided and confirmed done: **`336x602` on
     all three boards.** This note previously said R3 still ran the old `252x448` and needed
     updating — stale; checking R3's actual running container config
     (`docker inspect da3 --format '{{range .Config.Cmd}}{{.}} {{end}}'`) showed it was
     already on `DA3METRIC-LARGE.336x602.fp16.engine`. The old `L-252x448` folder (ONNX only,
     no engine) is leftover and unused on both R1 and R3 — safe to delete whenever (TODO
     item 7 territory).
   - Compare `DEPTH_ANYTHING` distances to known distances - is either size accurate enough?
     Saw some negative depth minimums in R2's debug log (e.g. `depth min=-0.632213`) -
     plausibly explained by the fisheye issue below rather than plain model noise: `cam.yaml`
     encodes a pinhole model, so peripheral/edge pixels (worst distortion) get a wrong
     pixel-to-ray mapping, which could produce exactly this kind of implausible depth.
   - Lens is a fisheye: calibrate (OpenCV fisheye) and undistort before DA3, then fix `cam.yaml`
     (this is also the real calibration this camera needs - the current one is just the 4K
     calibration divided by 6, never independently measured for this lens).
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
