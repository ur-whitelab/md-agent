from mdagent.general_tools import web_search


def test_web_search():
    result = web_search("example keywords", "google")
    # this will fail if our API key is not set
    assert result != "No results, try another search"