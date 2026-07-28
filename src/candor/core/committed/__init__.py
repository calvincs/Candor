"""The committed tier — the only place in the tree where weights live.

Spec §3.2 makes the I2 firewall *lexical*: nothing named "weight" exists
outside this package, and a grep for it elsewhere is an audit
(§6.2 "lexical firewall"). Keep it that way.
"""
