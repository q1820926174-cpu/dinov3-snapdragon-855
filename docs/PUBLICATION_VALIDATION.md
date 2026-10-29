# Publication validation — 2026-10-29

The public source copy was verified locally before its first push:

- Re-exported the authorized official model through the public Windows/POSIX-compatible PNNX CLI. PyTorch dense/global maximum absolute errors: 1.3828e-5 / 3.5763e-6. The resulting `model.param` and `model.bin` SHA256 values **exactly matched the model already measured on the phone**. Neither file is distributed.
- Compiled all three ARM64 executables from this public source directory with GNU 12 and the existing tested ncnn installation. A first configure attempt hit the local toolchain's Make/Ninja mismatch; configuring a separate build directory with `-G Ninja` resolved it. No change to application source was required. This does not claim a fresh third-party dependency build or a new full device benchmark.
- Checked `ncnn-scv41.patch` against clean source files from the exact pinned upstream commit. The preexisting dependency checkout was not modified.
- Reconstructed all three gallery images and four original application cases from the fixed source metadata, checking source SHA256. Reconstructed gallery pixels matched the originally measured crops. Images remain Git-ignored and unpublished.
- Recomputed system power and request-integrated energy from the sanitized public raw telemetry. Results match the stated 3.62/4.64 W and 53.65/117.91 J per screenshot.
- Checked tracked files for prohibited artifacts, private absolute paths and common credential patterns. Python sources passed compilation. The public manifest hashes canonical Git-index bytes, excluding itself, so its values remain stable across filesystem newline settings.

Historical device records describe the originally measured code/artifacts. Model hashes and measurement values are preserved; private local paths, screen-process metadata, and unpublished image/weight content are omitted. No new model quality or hardware performance claim follows merely from publication checks.
