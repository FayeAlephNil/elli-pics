# Elli-pics

This repository will contain different pictures of Elliptic Fibrations. Currently it contains 3 relevant files

- poly.py
- main.py
- data_gen.ipynb

The file polys.py is essentially deprecated -- the method for computing the j invariant had too many issues.

The data_gen Jupyter notebook contains Sage code for computing the singular locus of a rational elliptic fibration defined via a pencil of cubic curves sP + tQ. It also allows one to output this data as P varies to a file for use by the rest of the program.

main.py uses Manim's animation engine to produce short moves. The relevant one to data_gen is AnimatePointsOnSphere. Selecting this with Manim will read the relevant file and produce a moving picture of the relevant divisors produced by data_gen.
