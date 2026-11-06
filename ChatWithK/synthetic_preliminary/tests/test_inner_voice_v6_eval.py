from synthetic_preliminary.eval_inner_voice_v6_annotations import token_boundary, word_cut_char


def test_word_cut_and_token_alignment():
    text = "I am useless person"
    assert word_cut_char(text, 3) == len("I am useless")
    offsets = [(0, 1), (2, 4), (5, 12), (13, 19)]
    assert token_boundary(offsets, 12) == (3, True)
    assert token_boundary(offsets, 11) == (2, False)
