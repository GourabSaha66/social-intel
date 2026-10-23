import networkx as nx
import pandas as pd
import pytest

from socintel import graph


def test_edges_skip_self_deleted_and_unknown_parents(thread_df):
    e = graph.build_edges(thread_df)
    pairs = {(r.src, r.dst): r.weight for r in e.itertuples()}
    assert pairs[("bob", "alice")] == 1
    assert pairs[("carol", "alice")] == 1
    assert pairs[("dave", "bob")] == 1
    assert pairs[("bob", "dave")] == 1
    assert ("bob", "bob") not in pairs                  # self reply dropped
    assert not any("[deleted]" in p for p in pairs)     # deleted author dropped
    assert not any("eve" in p for p in pairs)           # parent never seen


def test_repeated_replies_add_weight():
    df = pd.DataFrame({
        "post_id": ["p", "a", "b"], "author": ["x", "y", "y"], "parent_id": [None, "p", "p"]})
    e = graph.build_edges(df)
    assert e.loc[0, "weight"] == 2


def test_pagerank_ranks_the_hub_first():
    # everyone replies to "hub"
    edges = pd.DataFrame({"src": ["a", "b", "c", "d", "hub"], "dst": ["hub", "hub", "hub", "hub", "a"],
                          "weight": [1, 1, 1, 1, 1]})
    g = graph.build_graph(edges)
    ranks = graph.rank_users(g)
    assert ranks.iloc[0]["user"] == "hub"
    assert ranks["pagerank"].sum() == pytest.approx(1.0)


def test_communities_split_two_clusters():
    e = []
    for grp in (["a1", "a2", "a3", "a4"], ["b1", "b2", "b3", "b4"]):
        for i in grp:
            for j in grp:
                if i != j:
                    e.append((i, j, 3))
    e.append(("a1", "b1", 1))                          # one weak bridge
    g = graph.build_graph(pd.DataFrame(e, columns=["src", "dst", "weight"]))
    ranks = graph.rank_users(g)
    comm = ranks.set_index("user")["community"]
    assert comm["a1"] == comm["a2"] == comm["a3"]
    assert comm["b1"] == comm["b2"] == comm["b3"]
    assert comm["a1"] != comm["b1"]
    assert graph.modularity(g, ranks) > 0.3


def test_empty_graph_does_not_crash():
    g = nx.DiGraph()
    assert graph.rank_users(g).empty
    assert graph.graph_summary(g)["nodes"] == 0
