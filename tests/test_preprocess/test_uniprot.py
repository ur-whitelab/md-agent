from mdagent.tools.base_tools.preprocess_tools.pdb_tools.uniprot import UniProt_Converter


def test_uniprot_converter():
    converter = UniProt_Converter()
    # Example usage
    # # Example usage
    uniprot_id = "P12345"
    to_db = "KEGG"
    assert converter.convert(uniprot_id, to_db).strip() == "Associated KEGG ids: ocu:100348732"

    to_db = "nonsese"
    assert "Error - one of the input parameters is incorrect" in converter.convert(uniprot_id, to_db)

    to_db = "PDB"
    pdb_output = converter.convert("p68871", to_db)
    msg, pdb_ids = pdb_output.split(":")
    assert msg.strip() == "Associated PDB ids"
    #convert pdb_ids to list
    pdb_ids = pdb_ids.strip().split(", ")
    print (type(pdb_ids))
    assert set(pdb_ids) == set(["1A00", "1A01", "1A0U", "1A0Z", "1A3N", "1A3O", "1ABW", "1ABY", "1AJ9", "1B86", "1BAB", "1BBB", "1BIJ", "1BUW", "1BZ0", "1BZ1", "1BZZ", "1C7B", "1C7C", "1C7D", "1CBL", "1CBM", "1CH4", "1CLS", "1CMY"])