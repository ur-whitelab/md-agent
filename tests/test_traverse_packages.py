from mdagent.tools.traverse_packages import TraversePackages


def test_list_attributes():
    tp = TraversePackages()
    all_attributes = tp.list_attributes("numpy")
    assert "vectorize" in all_attributes.keys()


def test_help_all_attributes():
    tp = TraversePackages()
    all_attribute_details = tp.help_all_attributes("mdtraj")
    assert type(all_attribute_details) == dict
    assert (
        "A string is alpha-numeric if all characters in the string are alpha-numeric"
        in str(all_attribute_details.values())
    )


def test_import_module():
    tp = TraversePackages()
    module = tp.import_module("mdtraj")
    assert module.__name__ == "mdtraj"


def test_help_on_attribute():
    tp = TraversePackages()
    help_on_attribute = tp.help_on_attribute("mdtraj", "compute_rg")
    assert "Help on function compute_rg in module mdtraj.geometry" in help_on_attribute

    help_on_attribute = tp.help_on_attribute("matplotlib.pyplot", "plot")
    assert "Help on function plot in module matplotlib.pyplot" in help_on_attribute
