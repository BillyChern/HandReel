# E1 amendment (infrastructure only): disk floor 175 → 150 GB

Registered before any E1 held-out output exists: no unit of the final queue has started, nothing is sealed, and no
outcome has been read. Written 2026-10-06T17:53:13Z.

- **What changes:** `run/study/e1_common.py` `DISK_FLOOR_GB` goes from 175 to 150. A unit now starts only if free disk minus the
  15 GB unit reservation stays ≥ 150 GB, i.e. at ≥ 165 GB free (prereg s. 9.3 / 10 text "≥ 175 GB" reads "≥ 150 GB").
- **Why:** other tenants of the shared volume pulled free space from 302 GB (2026-10-05) to 171 GB
  (2026-10-06T17:50Z), below the old start threshold of 190 GB. Each E1 unit keeps ~0.5 MB after packing; its real
  need is the transient working set covered by the unchanged 15 GB reservation.
- **Authority:** amendment approved by the authors on 2026-10-06T17:50Z (lower the floor to 150 GB).
- **Effect on outcomes:** none. The floor only gates WHEN a unit may start; units, order, code paths, scorer and
  thresholds are unchanged. If free disk falls below 165 GB the queue waits, as before.
