import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def thread_df():
    """tiny reddit-like thread: alice posts, bob/carol reply, dave replies to bob, eve spams."""
    rows = [
        ("r_p1", "post", "alice", 1_700_000_000, "Big news about the vaccine rollout", None),
        ("r_c1", "comment", "bob", 1_700_000_100, "this is great news", "r_p1"),
        ("r_c2", "comment", "carol", 1_700_000_200, "I do not trust this at all", "r_p1"),
        ("r_c3", "comment", "dave", 1_700_000_300, "agree with bob", "r_c1"),
        ("r_c4", "comment", "bob", 1_700_000_400, "thanks dave", "r_c3"),
        ("r_c5", "comment", "bob", 1_700_000_500, "talking to myself", "r_c1"),   # self reply
        ("r_c6", "comment", "[deleted]", 1_700_000_600, "gone", "r_p1"),
        ("r_c7", "comment", "eve", 1_700_000_700, "reply to a post we never saw", "r_zzz"),
    ]
    return pd.DataFrame(rows, columns=["post_id", "kind", "author", "created_utc", "text", "parent_id"])
