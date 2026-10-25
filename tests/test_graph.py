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


def test_replies_only_drops_edges_to_posts(thread_df):
    full = {(r.src, r.dst) for r in graph.build_edges(thread_df).itertuples()}
    rep = {(r.src, r.dst) for r in graph.build_edges(thread_df, replies_only=True).itertuples()}
    assert ("bob", "alice") in full and ("carol", "alice") in full      # comments on alice's post
    assert ("bob", "alice") not in rep and ("carol", "alice") not in rep
    assert rep == {("dave", "bob"), ("bob", "dave")}                     # comment-to-comment only


def test_video_owner_tops_full_graph_but_not_replies_graph():
    rows = [("p", "post", "owner", None)]
    rows += [(f"c{i}", "comment", f"u{i}", "p") for i in range(6)]       # everyone comments on the video
    rows += [("r1", "comment", "u1", "c2"), ("r2", "comment", "u3", "c2"),
             ("r3", "comment", "u4", "c2")]                              # three people answer u2
    df = pd.DataFrame(rows, columns=["post_id", "kind", "author", "parent_id"])
    full = graph.rank_users(graph.build_graph(graph.build_edges(df)))
    rep = graph.rank_users(graph.build_graph(graph.build_edges(df, replies_only=True)))
    assert full.iloc[0]["user"] == "owner"
    assert rep.iloc[0]["user"] == "u2"


def test_graph_summary_reports_components():
    g = nx.DiGraph([("a", "b"), ("c", "d"), ("d", "e")])
    s = graph.graph_summary(g)
    assert s["n_components"] == 2 and s["largest_component_nodes"] == 3
    assert s["largest_component_share"] == pytest.approx(0.6)
    assert graph.graph_summary(nx.DiGraph())["n_components"] == 0