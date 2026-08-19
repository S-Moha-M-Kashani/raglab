"""The lab's terminal tools — the four modules no frontend route reaches.
Each is a self-contained action over the lab's records (run a sweep, screen a judge, rank the runs, export a report), grouped here so they can later be offered as tools to an LLM on the frontend.
Everything in this package depends on the lab (`raglab.*`); nothing in the lab depends on it — the dependency points inward, the `service_route_plumbing.py` rule.
"""
