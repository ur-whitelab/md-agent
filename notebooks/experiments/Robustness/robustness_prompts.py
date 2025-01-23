descriptive_prompts = ["Complete all of the following tasks: 1. Download the PDB file 1LYZ.",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the structure.",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Simulate for 1 ps at 300 K.",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Simulate for 1 ps at 300 K. 4. Compute the RMSD.",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Simulate for 1 ps at 300 K. 4. Compute the RMSD. 5. Compute the radius of gyration over time.",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Simulate for 1 ps at 300 K. 4. Compute the RMSD. 5. Compute the radius of gyration over time. 6. Compute the SASA (solvent accessible surface area).",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Find any known binding or active sites. 4. Simulate for 1 ps at 300 K. 5. Compute the RMSD. 6. Compute the radius of gyration over time. 7. Compute the SASA (solvent accessible surface area).",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Simulate for 1 ps at 300 K. 4. Compute the RMSD of the simulation at 300 K. 5. Compute the radius of gyration over time for the simulation at 300 K. 6. Simulate for 1 ps at 400 K. 7. Compute the RMSD of the simulation at 400 K. 8. Compute the radius of gyration over time for the simulation at 400 K.",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Find any known binding or active sites. 4. Simulate for 1 ps at 300 K. 5. Compute the RMSD of the simulation at 300 K. 6. Compute the radius of gyration over time for the simulation at 300 K. 7. Simulate for 1 ps at 400 K. 8. Compute the RMSD of the simulation at 400 K. 9. Compute the radius of gyration over time for the simulation at 400 K.",
"Complete all of the following tasks: 1. Download the PDB file 1LYZ. 2. Report the secondary structure assignments of the PDB structure. 3. Simulate for 1 ps at 300 K. 4. Compute the RMSD of the simulation at 300 K. 5. Compute the radius of gyration over time for the simulation at 300 K. 6. Compute the SASA (solvent accessible surface area) for the simulation at 300 K. 7. Simulate for 1 ps at 400 K. 8. Compute the RMSD of the simulation at 400 K. 9. Compute the radius of gyration over time for the simulation at 400 K. 10. Compute the SASA (solvent accessible surface area) for the simulation at 400 K."]

natural_prompts = ["Download the PDB file 1LYZ.",
"Download the PDB file 1LYZ. Report the secondary structure assignments of the structure.",
"Simulate 1LYZ for 1 ps at 300 K. Report the secondary structure assignments of the PDB structure.",
"Simulate 1LYZ for 1 ps at 300 K. Report the secondary structure assignments of the PDB structure, and compute the RMSD of the simulation.",
"Simulate 1LYZ for 1 ps at 300 K. Report the secondary structure assignments of the PDB structure, and compute the RMSD and radius of gyration of the simulation.",
"Simulate 1LYZ for 1 ps at 300 K. Report the secondary structure assignments of the PDB structure, and compute the RMSD, SASA, and radius of gyration of the simulation.",
"Simulate 1LYZ for 1 ps at 300 K. Report the secondary structure assignments of the PDB structure and any known binding or active sites. Compute the RMSD, SASA, and radius of gyration of the simulation.", 
"Simulate 1LYZ for 1 ps at 300K and 400K. Report the secondary structure assignments of the PDB structure, and compute the RMSD and radius of gyration of both simulations.",
"Simulate 1LYZ for 1 ps at 300K and 400K. Report the secondary structure assignments of the PDB structure and any known binding or active sites. Compute RMSD and radius of gyration of both simulations.",
"Simulate 1LYZ for 1 ps at 300K and 400K. Report the secondary structure assignments of the PDB structure, and compute the RMSD, SASA, and radius of gyration of both simulations."]

def get_prompt(descriptive_or_natural, num):
    if descriptive_or_natural == "descriptive":
        return descriptive_prompts[num-1]
    elif descriptive_or_natural == "natural":
        return natural_prompts[num-1]
    