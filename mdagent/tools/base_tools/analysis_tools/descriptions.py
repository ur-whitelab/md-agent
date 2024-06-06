TRAJECTORY_FILEID_DESC = "Trajectory File ID of the simulation to be analyzed"

TOPOLOGY_FILEID_DESC = "Topology File ID of the simulation to be analyzed"

SELECTION_DESC = "Selection of atoms from the simulation to use for the analysis"

RES_SELECTION_DESC = (
    "Selection of residues ids from the simulation to use for the analysis."
    "Example selection: 'resid 0 to 10' or 'resid 0 1 2 3 4 5 6 7 8 9 10'"
)
CONTACT_SELECTION_DESC = "Selection of residues from the simulation to use for the \
    contact analysis. Default is 'all' which will calculate the distance between all\
          residue pairs.\nExample selection: 'resid 0 to 10' or \
                'resid 0 1 2 3 4 5 6 7 8 9 10' or 'all'"

CUTOFF_DESC = "Hard cutoff distance for the contact analysis in nanometers.\
    Defaults to 0.8"

#####################Tool Descriptions#####################
DISTANCE_TOOL_DESC = (
    "Tool for calculating distances between residue pairs in each frame of a"
    " trajectory. If only one pair is provided, the tool will calculate the distance"
    " between said pair in each frame and output a distance vs time plot and a "
    " histogram. iF multiple pairs are provided, the tool will calculate the distance "
    "between each pair in each frame and output a distance matrix plot for the selected"
    " pairs.\n You can use 'analysis' = 'all' to calculate the distance between all "
    "residue pairs in each frame. Or if interested in a specific pair, you can provide "
    "two selections of residues/atoms to calculate the distance between them."
)

DISPLACEMENT_TOOL_DESC = (
    "Tool for calculating displacement vectors"
    " between atom selections pairs in a trajectory."
)
NEIGHBORS_TOOL_DESC = (
    "Find (spatially) neighboring atoms in a trajectory.\n\n"
    "Given a selection of atoms and a distance cutoff, computes the indices of all"
    " atoms whose distance to 1 or more of the query points is less than cutoff."
)

CONTACTS_TOOL_DESC = (
    "Tool for computing the distance between pairs of residues in a trajectory. "
    "If distance is under the cutoff is considered a contact. The output is a matrix "
    "plot where each contact between residues is represented by a dot."
)
