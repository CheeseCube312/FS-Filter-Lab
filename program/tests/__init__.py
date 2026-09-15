"""FS FilterLab behavior suite.

Run from the program/ directory:

    python -m unittest discover -s tests -v

Behavior-only regression net: every test names a catchable mistake, uses
hand-computed oracles on toy grids where physics is involved, and never
reimplements module internals. See tests/helpers.py for shared builders.
"""