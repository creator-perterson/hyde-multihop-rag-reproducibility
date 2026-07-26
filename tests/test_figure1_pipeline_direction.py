from pathlib import Path


FIGURE_SOURCE = Path(__file__).resolve().parents[1] / "paper/latex" / "figure1_method_pipeline_tikz.tex"


def test_figure1_main_flow_uses_explicit_left_to_right_anchors():
    source = FIGURE_SOURCE.read_text(encoding="utf-8")
    expected_edges = [
        r"\draw[flowarrow] (q.east) -- (gen.west);",
        r"\draw[flowarrow] (gen.east) -- (hyp.west);",
        r"\draw[flowarrow] (hyp.east) -- (query.west);",
        r"\draw[flowarrow] (query.east) -- (retr.west);",
        r"\draw[flowarrow] (retr.east) -- (evid.west);",
        r"\draw[flowarrow] (evid.east) -- (reader.west);",
        r"\draw[flowarrow] (reader.east) -- (ans.west);",
    ]

    for edge in expected_edges:
        assert edge in source


def test_figure1_main_flow_does_not_use_ambiguous_center_to_center_arrows():
    source = FIGURE_SOURCE.read_text(encoding="utf-8")
    ambiguous_edges = [
        r"\draw[arrow] (gen) -- (hyp);",
        r"\draw[arrow] (hyp) -- (query);",
        r"\draw[arrow] (query) -- (retr);",
        r"\draw[arrow] (retr) -- (evid);",
    ]

    for edge in ambiguous_edges:
        assert edge not in source
