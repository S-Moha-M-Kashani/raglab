"""The lab's terminal tools — the three modules no frontend route reaches.
Each is a self-contained action over the lab's records (run a sweep, screen a judge, export a report), grouped here so they can later be offered as tools to an LLM on the frontend.
`leaderboard.py` used to be the fourth. It moved to `raglab.evaluation` when the panel grew a leaderboard surface: it reads what the runs recorded and ranks them, which is scoring's business, and a route may reach it there without making this package's rule false.
Everything in this package depends on the lab (`raglab.*`); nothing in the lab depends on it — the dependency points inward, the `service_route_plumbing.py` rule.
"""
