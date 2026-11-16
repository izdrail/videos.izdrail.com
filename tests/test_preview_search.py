from core.media.preview_search import preview_query,youtube_relevant

def test_script_subject_retained():
    assert preview_query('A river flows through a green forest.','green forest')=='river green forest'

def test_explicit_query_wins():
    assert preview_query('A river flows.','river','mountain waterfall')=='mountain waterfall'

def test_unrelated_channel_match_held():
    q=preview_query('A river flows through a green forest.','green forest')
    assert not youtube_relevant('Robert and Mai raise pigs. Green forest life',q)
    assert not youtube_relevant('Building a road to a green forest farm',q)
    assert youtube_relevant('River running through a green forest - drone footage',q)
    assert youtube_relevant('Forest river aerial footage',q)

def test_missing_title_not_called_relevant():
    assert not youtube_relevant('', 'river forest')

def test_word_boundaries_not_substrings():
    assert not youtube_relevant('Driver forest documentary','river forest')

def test_unicode_query_is_preserved():
    assert preview_query('Râul trece prin pădure','pădure').startswith('râul')

def test_short_explicit_subject():
    assert youtube_relevant('Waterfall drone shot','waterfall')
