"""Trusted core (spec §1). Standard library only, no periphery imports, ever.

Deliberately empty so that importing any single core module never drags the
count updater into the import graph of an unrelated component (I2).
"""
