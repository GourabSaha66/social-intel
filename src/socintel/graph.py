"""User-interaction graph: edge A -> B means "A replied to B".

NetworkX does the maths (PageRank, communities); Neo4j stores the graph so it can be
explored visually and queried with Cypher. Edge weight = number of replies.
"""

import os
from collections import Counter

import networkx as nx
import pandas as pd

SKIP_AUTHORS = {"[deleted]", "AutoModerator", "unknown", "None"}


def build_edges(df):
    """replier -> parent author, using parent_id lookups inside the table."""
    author_of = dict(zip(df["post_id"], df["author"]))
    pairs = Counter()
    for pid, author in zip(df["parent_id"], df["author"]):
        if pid is None or pd.isna(pid):
            continue
        parent_author = author_of.get(pid)
        if parent_author is None or parent_author == author:       # unknown parent / self reply
            continue
        if author in SKIP_AUTHORS or parent_author in SKIP_AUTHORS:
            continue
        pairs[(author, parent_author)] += 1
    return pd.DataFrame([(a, b, w) for (a, b), w in pairs.items()],
                        columns=["src", "dst", "weight"])


def build_graph(edges):
    g = nx.DiGraph()
    for r in edges.itertuples(index=False):
        g.add_edge(r.src, r.dst, weight=r.weight)
    return g


def rank_users(g, seed=42):
    """PageRank (weighted, directed) + Louvain communities on the undirected projection."""
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["user", "pagerank", "in_degree", "out_degree", "community"])
    pr = nx.pagerank(g, alpha=0.85, weight="weight")
    comms = nx.community.louvain_communities(g.to_undirected(), weight="weight", seed=seed)
    comm_of = {u: i for i, c in enumerate(sorted(comms, key=len, reverse=True)) for u in c}
    out = pd.DataFrame({
        "user": list(g.nodes),
        "pagerank": [pr[u] for u in g.nodes],
        "in_degree": [g.in_degree(u, weight="weight") for u in g.nodes],
        "out_degree": [g.out_degree(u, weight="weight") for u in g.nodes],
        "community": [comm_of[u] for u in g.nodes],
    })
    return out.sort_values("pagerank", ascending=False).reset_index(drop=True)


def snap_edges(path, nrows=None):
    """SNAP Reddit Hyperlink Network (soc-redditHyperlinks-title.tsv / -body.tsv).
    Columns: SOURCE_SUBREDDIT, TARGET_SUBREDDIT, POST_ID, TIMESTAMP, LINK_SENTIMENT, PROPERTIES.
    Nodes are subreddits (not people). weight = number of hyperlinks, sent = mean link sentiment
    (+1 positive / -1 negative). Published academic dataset from Stanford SNAP.
    """
    d = pd.read_csv(path, sep="\t", nrows=nrows,
                    usecols=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT"])
    d = d[d["SOURCE_SUBREDDIT"] != d["TARGET_SUBREDDIT"]]
    e = (d.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"])
          .agg(weight=("LINK_SENTIMENT", "size"), sent=("LINK_SENTIMENT", "mean")).reset_index())
    return e.rename(columns={"SOURCE_SUBREDDIT": "src", "TARGET_SUBREDDIT": "dst"})


def graph_summary(g):
    und = g.to_undirected()
    biggest = max(nx.connected_components(und), key=len) if und.number_of_nodes() else set()
    return {
        "nodes": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "density": nx.density(g),
        "largest_component_share": len(biggest) / max(g.number_of_nodes(), 1),
    }


def modularity(g, ranks):
    groups = ranks.groupby("community")["user"].apply(set).tolist()
    return nx.community.modularity(g.to_undirected(), groups, weight="weight")


# ---------------- Neo4j ----------------

def neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                                auth=(os.environ.get("NEO4J_USER", "neo4j"),
                                      os.environ.get("NEO4J_PASSWORD", "changeme123")))


def load_into_neo4j(edges, ranks, driver, batch=5000):
    with driver.session() as s:
        s.run("CREATE CONSTRAINT user_name IF NOT EXISTS FOR (u:User) REQUIRE u.name IS UNIQUE")
        s.run("MATCH (n) DETACH DELETE n")          # full reload each run, fine at this size
        rows = ranks.to_dict("records")
        for i in range(0, len(rows), batch):
            s.run("UNWIND $rows AS r MERGE (u:User {name: r.user}) "
                  "SET u.pagerank = r.pagerank, u.community = r.community, "
                  "u.in_degree = r.in_degree, u.out_degree = r.out_degree",
                  rows=rows[i:i + batch])
        er = edges.to_dict("records")
        for i in range(0, len(er), batch):
            s.run("UNWIND $rows AS r "
                  "MATCH (a:User {name: r.src}), (b:User {name: r.dst}) "
                  "MERGE (a)-[e:REPLIED_TO]->(b) SET e.weight = r.weight",
                  rows=er[i:i + batch])


TOP_INFLUENCERS_CYPHER = """
MATCH (u:User) RETURN u.name AS user, u.pagerank AS pagerank, u.community AS community
ORDER BY u.pagerank DESC LIMIT $n
"""

COMMUNITY_SIZES_CYPHER = """
MATCH (u:User) RETURN u.community AS community, count(*) AS members,
       avg(u.pagerank) AS avg_pagerank ORDER BY members DESC
"""
