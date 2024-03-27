# Second set of draft codes:

    import mdtraj as md


def find_salt_bridges_alternative(traj, threshold_distance=0.4, residue_pairs=None):
    # Salt bridge analysis implementation
    salt_bridges = []
    if residue_pairs is None:
        residue_pairs = [("ARG", "ASP"), ("ARG", "GLU"), ("LYS", "ASP"), ("LYS", "GLU")]

    for pair in residue_pairs:
        donor_residues = traj.topology.select(f'residue_name == "{pair[0]}"')
        acceptor_residues = traj.topology.select(f'residue_name == "{pair[1]}"')

        for donor_idx in donor_residues:
            for acceptor_idx in acceptor_residues:
                distances = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
                if any(d <= threshold_distance for d in distances):
                    salt_bridges.append((donor_idx, acceptor_idx))

    return salt_bridges


def main():
    # Define file paths
    trajectory_file = "trajectory.dcd"
    topology_file = "topology.pdb"

    # Load trajectory using MDTraj
    traj = md.load(trajectory_file, top=topology_file)

    # Parameters for salt bridge analysis
    threshold_distance = 0.4  # Maximum distance between charged atoms
    residue_pairs = [
        ("ARG", "ASP"),
        ("ARG", "GLU"),
        ("LYS", "ASP"),
        ("LYS", "GLU"),
    ]  # Residue pairs to consider

    # Perform salt bridge analysis
    salt_bridges = find_salt_bridges(
        traj, threshold_distance=threshold_distance, residue_pairs=residue_pairs
    )

    # Print identified salt bridges
    print("Salt bridges found:")
    for bridge in salt_bridges:
        print(
            f"Residue {traj.topology.atom(bridge[0]).residue.index + 1} ({traj.topology.atom(bridge[0]).residue.name}) - "
            f"Residue {traj.topology.atom(bridge[1]).residue.index + 1} ({traj.topology.atom(bridge[1]).residue.name})"
        )


if __name__ == "__main__":
    main()
