# Decision log

Record protocol amendments, failures, and schedule decisions here as they occur.

- 2026-09-11T10:21:42.146558+00:00: Interrupted the first 4B preflight after approximately 40 minutes before any preflight artifact or scientific evaluation was produced. The full-parameter SHA-256 implementation was converting Torch storage through Python byte iteration. Replaced only the byte transport with an equivalent buffer-backed SHA-256 update. Exact-byte equivalence and parameter-mutation sensitivity tests passed before rerunning.
