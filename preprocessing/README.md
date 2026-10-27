# Preprocessing code (archival)

This directory contains the original research code that converts the raw head and gaze `.csv` logs into the `.pck`
files distributed with the dataset. It was copied unchanged from our internal development repository (last commit
from May 2017) and is released **as-is** for provenance. It is not maintained.

Caveats:

* The code is Python 2.7 (`cPickle`, `xrange`, `iteritems`, integer division) and will not run under Python 3 without
  modification.
* Paths are hard-coded for the original development machines: see `settings.py` and the `__main__` block of
  `preprocessing.py`. `settings.py` here is the one the preprocessing code was written against; it differs from
  `src/settings.py`, which is used by the analysis notebook.
* We cannot guarantee that this snapshot is byte-for-byte the version that generated every released `.pck` file.

## Files

* `preprocessing.py`: the pipeline. Entry point is `consolidate_log_to_dfs(experiment_dir, out_dir, ...)`, which
  expects one sub-directory per panorama containing `*head*.csv` / `*gaze*.csv` log pairs and writes one
  `<panorama>.pck` (a pickled `pandas.DataFrame`, one row per run) per sub-directory.
* `fixation_detector.py`: dispersion/duration fixation detector adapted from Pupil Labs (LGPL v3.0, see the header).
* `data_analysis.py`: analysis helpers; `preprocessing.py` imports it for `lat_long_to_yx` and plotting.
* `settings.py`: paths used by the code above.

## Pipeline summary

For each head/gaze log pair, `consolidate_log_to_dfs` does the following:

1. **Parse** both logs (`parse_head_logfile`, `parse_gaze_logfile`). Rows whose vector fields do not have the expected
   shape are dropped.
2. **Combine eyes** (`join_left_right_gaze`). The Pupil Labs log has separate rows for the left and right eye. The
   stream with more samples is used as the time base and each of its samples is paired with the nearest-in-time sample
   of the other eye. Normalized coordinates (`both_coords`) and confidences (`both_confidence`) are the mean of the two
   eyes.
3. **Clean gaze** (`clean_gaze_records`), in normalized eye-tracker coordinates, before any projection:
   * leading rows recorded before the timestamp reset (timestamp >= 1.0) are dropped;
   * samples with a coordinate outside [-0.5, 1.5] get confidence 0;
   * samples with confidence below 0.8 are replaced by **nearest-neighbour** interpolation from the high-confidence
     samples; low-confidence samples before the first or after the last high-confidence sample are set to the mean
     high-confidence position.
4. **Discard runs** whose mean confidence (after step 3) is below `mean_conf_thresh` (0.7 in the `__main__` call).
5. **Head velocity/acceleration** is computed on the raw head stream (`get_ang_va`), with wrap-around jumps
   interpolated.
6. **Align head and gaze** (`get_closest_timestamp_merge`). Again the stream with more samples is the time base and the
   other is sampled by nearest timestamp. Head orientation is **not** interpolated (no slerp); it is nearest-neighbour
   matched.
7. **Remap head orientation**: `headLatLon = mod((headLatLon_raw + [90, 180]) * [-1, 1], [180, 360])`, i.e. latitude in
   [0, 180] and longitude in [0, 360). The head orientation used is the logged (lat, lon) pair plus the logged
   `headTilt` (roll), not the quaternion.
8. **Project gaze to the sphere** (`get_gaze_lat_lon`):
   * head roll is applied as a 2D rotation of the normalized gaze point about the viewport centre (0.5, 0.5) by
     `headTilt` degrees (`gaze_account_head_tilt`);
   * y is flipped (`y = 1 - y`, Pupil Labs' origin is bottom-left);
   * the point is back-projected with a pinhole model using the Unity/Oculus DK2 field of view,
     `fov_v = 106.18877` deg and `fov_h = 0.82034051 * fov_v` (~87.1 deg);
   * the ray is rotated by `R_lon @ R_lat`, where `R_lat` rotates about x by `-lat` and `R_lon` about z by `-lon` of
     the (remapped) head orientation;
   * `lat = degrees(arccos(z))`, `lon = degrees(arctan2(x, y))`, wrapped to [0, 180) x [0, 360).

   This is the same construction as `gnomonic2lat_lon()` in `src/Analysis.ipynb`, but with per-sample head rotation
   and the roll step.
9. **Smooth**: latitude and longitude of `gaze_lat_lon` are each filtered independently with a 4-tap box filter
   (`np.convolve(..., mode='same')`). The filter is applied to the wrapped values and zero-padded at the ends, so it
   distorts samples next to the 0/360 longitude seam, near the poles, and at the first and last samples of a run.
10. **Gaze-in-head offset**: `gaze_lat_lon_offset = gaze_lat_lon - headLatLon`; longitude offsets >= 100 or <= -100 are
    shifted by 360, and any remaining samples with |longitude offset| > 100 deg are linearly interpolated.
11. **Fixations** (`get_eye_fixations`) via the dispersion/duration detector on equirectangular-normalized gaze, then
    gaze velocity and acceleration (`get_ang_va`) for `gaze_lat_lon` and `gaze_lat_lon_offset`.
