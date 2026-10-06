# Tiny synthetic valley used to test the workflow without research data.
#
#   python examples/synthetic_valley/make_synthetic_valley.py
#   python -m avulsionprecursors extract -c examples/synthetic_valley/config.yaml
#   python examples/synthetic_valley/write_labels.py
#   python -m avulsionprecursors lambda  -c examples/synthetic_valley/config.yaml
#
# The DEM is created locally (a few tens of KB). It is a straight channel with
# 5 m levees, a 2 m floodplain step, and 40 m valley walls.
