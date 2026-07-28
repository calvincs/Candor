"""Untrusted periphery (spec §1). Fallible, always gated, always attributed.

Nothing here may write to the committed tier. `retrieval.py` in particular must
keep an empty import path to the count updater; the audit is mechanical
(§6.2 count provenance) and lives in `candor.audit`.
"""
