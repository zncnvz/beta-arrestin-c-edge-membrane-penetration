"""
C-edge membrane penetration analysis for beta-arrestin1 and beta-arrestin2.

This script calculates residue-level membrane penetration for selected
C-edge regions from coarse-grained Martini 2 molecular dynamics trajectories.

Analyzed regions:
    344-Loop : residues 331-338
    197-Loop : residues 190-193

Membrane penetration is calculated relative to the average Z position
of phosphate (PO4) beads in the corresponding membrane leaflet.

Penetration:
    0 Å  = no membrane penetration
    >0 Å = membrane penetration
"""


# ============================================================
# IMPORTS
# ============================================================

import os

import MDAnalysis as mda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================
#
# Change the file names/paths below according to your system.
#
# In the original dataset:
#
# arr2 files correspond to beta-arrestin1
# arr3 files correspond to beta-arrestin2
#
# ============================================================


# ------------------------------------------------------------
# beta-arrestin1
# ------------------------------------------------------------

arr1_topology = "basal_arr2_membrane.pdb"

arr1_trajectory = "basal_arr2_membrane.xtc"


# ------------------------------------------------------------
# beta-arrestin2
# ------------------------------------------------------------

arr2_topology = "basal_arr3_membrane.pdb"

arr2_trajectory = "basal_arr3_membrane.xtc"


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

output_directory = "output"


# ============================================================
# SYSTEM DEFINITIONS
# ============================================================

systems = {

    "βarr1": {

        "topology":
            arr1_topology,

        "trajectory":
            arr1_trajectory,

    },

    "βarr2": {

        "topology":
            arr2_topology,

        "trajectory":
            arr2_trajectory,

    }

}


# ============================================================
# C-EDGE RESIDUE RANGES
# ============================================================
#
# The same residue numbering is used for beta-arrestin1
# and beta-arrestin2 in this analysis.
#
# ============================================================

residue_ranges = {

    "344-Loop":
        range(331, 339),

    "197-Loop":
        range(190, 194)

}


# ============================================================
# THREE-LETTER -> ONE-LETTER AMINO ACID CODE
# ============================================================

aa_3to1 = {

    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"

}


# ============================================================
# MARTINI 2 SETTINGS
# ============================================================

# Phosphate bead used to define membrane leaflets

phosphorus_selection = "name PO4"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    output_directory,
    exist_ok=True
)


# ============================================================
# CHECK INPUT FILES
# ============================================================

for protein_label, files in systems.items():

    if not os.path.exists(
        files["topology"]
    ):

        raise FileNotFoundError(

            f"{protein_label} topology file not found:\n"
            f"{files['topology']}"

        )

    if not os.path.exists(
        files["trajectory"]
    ):

        raise FileNotFoundError(

            f"{protein_label} trajectory file not found:\n"
            f"{files['trajectory']}"

        )


# ============================================================
# MEMBRANE PENETRATION FUNCTION
# ============================================================

def calculate_penetration(
    protein_label,
    topology,
    trajectory
):

    """
    Calculate residue-level membrane penetration for selected
    C-edge residues.

    Side-chain beads are used whenever available.
    For residues without side-chain beads, the backbone BB bead
    is used as a fallback.

    Parameters
    ----------
    protein_label : str
        Protein name used in the output.

    topology : str
        Topology file.

    trajectory : str
        Molecular dynamics trajectory.

    Returns
    -------
    pandas.DataFrame
        Frame-by-frame membrane penetration measurements.
    """


    print("\n")
    print("=" * 70)
    print("ANALYZING:", protein_label)
    print("=" * 70)


    print("\nTopology:")
    print(topology)


    print("\nTrajectory:")
    print(trajectory)


    # ========================================================
    # LOAD SYSTEM
    # ========================================================

    u = mda.Universe(
        topology,
        trajectory
    )


    print(
        "\nNumber of atoms:",
        len(u.atoms)
    )


    print(
        "Number of frames:",
        len(u.trajectory)
    )


    # ========================================================
    # PO4 SELECTION
    # ========================================================

    po4 = u.select_atoms(
        phosphorus_selection
    )


    if len(po4) == 0:

        raise ValueError(

            f"{protein_label}: "
            "No PO4 beads were found."

        )


    print(
        "Number of PO4 beads:",
        len(po4)
    )


    # ========================================================
    # PREPARE RESIDUES
    # ========================================================

    residue_groups = {}


    for region, resids in residue_ranges.items():

        for resid in resids:


            # ------------------------------------------------
            # First use side-chain beads
            # ------------------------------------------------

            atoms = u.select_atoms(

                f"protein and resid {resid} "
                f"and not name BB"

            )


            bead_type = "sidechain"


            # ------------------------------------------------
            # If no side-chain bead exists, use BB
            # ------------------------------------------------

            if len(atoms) == 0:

                atoms = u.select_atoms(

                    f"protein and resid {resid} "
                    f"and name BB"

                )

                bead_type = "BB_fallback"


            # ------------------------------------------------
            # If residue itself is missing
            # ------------------------------------------------

            if len(atoms) == 0:

                print(

                    f"WARNING: {protein_label} "
                    f"{region} resid {resid}: "
                    "residue/bead not found."

                )

                continue


            residue_groups[
                (region, resid)
            ] = {

                "atoms":
                    atoms,

                "bead_type":
                    bead_type

            }


    print(

        "\nNumber of residues included:",
        len(residue_groups)

    )


    # ========================================================
    # ANALYSIS
    # ========================================================

    results = []


    for ts in u.trajectory:


        # ====================================================
        # PO4 Z COORDINATES
        #
        # MDAnalysis coordinates are in Angstrom.
        # ====================================================

        po4_z = po4.positions[:, 2]


        # ====================================================
        # MEMBRANE CENTER
        # ====================================================

        membrane_center_z = np.mean(
            po4_z
        )


        # ====================================================
        # UPPER AND LOWER LEAFLETS
        # ====================================================

        upper_po4 = po4_z[
            po4_z > membrane_center_z
        ]


        lower_po4 = po4_z[
            po4_z < membrane_center_z
        ]


        if (
            len(upper_po4) == 0
            or
            len(lower_po4) == 0
        ):

            raise ValueError(

                f"{protein_label} "
                f"Frame {ts.frame}: "
                "upper/lower leaflet could not be determined."

            )


        upper_po4_z = np.mean(
            upper_po4
        )


        lower_po4_z = np.mean(
            lower_po4
        )


        # ====================================================
        # ANALYZE EACH RESIDUE
        # ====================================================

        for (
            region,
            resid
        ), residue_info in residue_groups.items():


            atoms = residue_info[
                "atoms"
            ]


            bead_type = residue_info[
                "bead_type"
            ]


            # =================================================
            # RESIDUE Z COORDINATES
            # =================================================

            residue_z = atoms.positions[:, 2]


            residue_center_z = np.mean(
                residue_z
            )


            # =================================================
            # UPPER LEAFLET
            # =================================================

            if residue_center_z >= membrane_center_z:


                selected_leaflet = "upper"


                deepest_z = np.min(
                    residue_z
                )


                penetration_A = (

                    upper_po4_z
                    -
                    deepest_z

                )


            # =================================================
            # LOWER LEAFLET
            # =================================================

            else:


                selected_leaflet = "lower"


                deepest_z = np.max(
                    residue_z
                )


                penetration_A = (

                    deepest_z
                    -
                    lower_po4_z

                )


            # =================================================
            # NEGATIVE VALUE = NO PENETRATION
            # =================================================

            penetration_A = max(
                0.0,
                penetration_A
            )


            # =================================================
            # RESIDUE NAME
            # =================================================

            resname_3 = (

                atoms.residues[0]
                .resname
                .upper()

            )


            resname_1 = aa_3to1.get(

                resname_3,
                resname_3

            )


            # =================================================
            # SAVE RESULT
            # =================================================

            results.append({

                "protein":
                    protein_label,

                "frame":
                    ts.frame,

                "time_ps":
                    ts.time,

                "region":
                    region,

                "resid":
                    resid,

                "resname":
                    resname_3,

                "resname_single":
                    resname_1,

                "bead_type":
                    bead_type,

                "selected_leaflet":
                    selected_leaflet,

                "membrane_center_z_A":
                    membrane_center_z,

                "upper_PO4_z_A":
                    upper_po4_z,

                "lower_PO4_z_A":
                    lower_po4_z,

                "residue_center_z_A":
                    residue_center_z,

                "deepest_z_A":
                    deepest_z,

                "penetration_A":
                    penetration_A

            })


    return pd.DataFrame(
        results
    )


# ============================================================
# RUN BOTH SYSTEMS
# ============================================================

all_data = []


for protein_label, files in systems.items():


    df_temp = calculate_penetration(

        protein_label,

        files["topology"],

        files["trajectory"]

    )


    all_data.append(
        df_temp
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

df = pd.concat(

    all_data,

    ignore_index=True

)


print("\n")
print("=" * 70)
print("BOTH ANALYSES COMPLETED")
print("=" * 70)


print("\nFirst rows:")

print(

    df.head()
    .to_string(
        index=False
    )

)


# ============================================================
# SAVE FRAME-BY-FRAME RESULTS
# ============================================================

output_csv = os.path.join(

    output_directory,

    "barr1_barr2_c_edge_membrane_penetration_all_frames_A.csv"

)


df.to_csv(

    output_csv,

    index=False

)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

summary = (

    df.groupby(

        [
            "protein",
            "region",
            "resid",
            "resname",
            "resname_single"
        ]

    )["penetration_A"]

    .agg(

        mean_penetration_A="mean",

        std_penetration_A="std",

        min_penetration_A="min",

        max_penetration_A="max",

        median_penetration_A="median"

    )

    .reset_index()

)


summary_csv = os.path.join(

    output_directory,

    "barr1_barr2_c_edge_membrane_penetration_summary_A.csv"

)


summary.to_csv(

    summary_csv,

    index=False

)


print("\nSUMMARY:")

print(

    summary.to_string(
        index=False
    )

)


# ============================================================
# BEAD TYPE SUMMARY
# ============================================================
#
# Shows which residues were analyzed using side-chain beads
# and which residues required BB fallback.
#
# ============================================================

bead_summary = (

    df[

        [
            "protein",
            "region",
            "resid",
            "resname_single",
            "bead_type"
        ]

    ]

    .drop_duplicates()

    .sort_values(

        [
            "protein",
            "region",
            "resid"
        ]

    )

)


print("\nBEAD TYPE CHECK:")

print(

    bead_summary.to_string(
        index=False
    )

)


bead_summary_csv = os.path.join(

    output_directory,

    "barr1_barr2_c_edge_bead_type_check.csv"

)


bead_summary.to_csv(

    bead_summary_csv,

    index=False

)


# ============================================================
# PLOTS
# ============================================================
#
# beta-arrestin1 = black
# beta-arrestin2 = red
#
# ============================================================

plot_files = []


for region, desired_resids in residue_ranges.items():


    # ========================================================
    # REGION DATA
    # ========================================================

    plot_df = summary[

        summary["region"] == region

    ].copy()


    residues = list(
        desired_resids
    )


    x = np.arange(
        len(residues)
    )


    width = 0.38


    # ========================================================
    # beta-arrestin1
    # ========================================================

    arr1_df = (

        plot_df[

            plot_df["protein"]
            ==
            "βarr1"

        ]

        .set_index(
            "resid"
        )

        .reindex(
            residues
        )

    )


    # ========================================================
    # beta-arrestin2
    # ========================================================

    arr2_df = (

        plot_df[

            plot_df["protein"]
            ==
            "βarr2"

        ]

        .set_index(
            "resid"
        )

        .reindex(
            residues
        )

    )


    # ========================================================
    # X-AXIS LABELS
    # ========================================================

    labels = []


    for resid in residues:


        # Prefer beta-arrestin1 residue name

        if (

            resid in arr1_df.index

            and

            pd.notna(

                arr1_df.loc[
                    resid,
                    "resname_single"
                ]

            )

        ):

            resname = arr1_df.loc[
                resid,
                "resname_single"
            ]


        # Otherwise use beta-arrestin2 residue name

        elif (

            resid in arr2_df.index

            and

            pd.notna(

                arr2_df.loc[
                    resid,
                    "resname_single"
                ]

            )

        ):

            resname = arr2_df.loc[
                resid,
                "resname_single"
            ]


        # Missing from both

        else:

            resname = "?"


        labels.append(

            f"{resname}{resid}"

        )


    # ========================================================
    # FIGURE
    # ========================================================

    plt.figure(
        figsize=(8, 6)
    )


    # ========================================================
    # beta-arrestin1
    # ========================================================

    plt.bar(

        x - width / 2,

        arr1_df[
            "mean_penetration_A"
        ].fillna(0),

        width=width,

        color="black",

        alpha=0.85,

        label="βarr1"

    )


    # ========================================================
    # beta-arrestin2
    # ========================================================

    plt.bar(

        x + width / 2,

        arr2_df[
            "mean_penetration_A"
        ].fillna(0),

        width=width,

        color="red",

        alpha=0.85,

        label="βarr2"

    )


    # ========================================================
    # ZERO LINE
    # ========================================================

    plt.axhline(

        0,

        color="gray",

        linestyle="--",

        linewidth=1.2

    )


    # ========================================================
    # X TICKS
    # ========================================================

    plt.xticks(

        x,

        labels,

        rotation=45,

        ha="right",

        fontsize=13,

        fontweight="bold"

    )


    # ========================================================
    # Y TICKS
    # ========================================================

    plt.yticks(

        fontsize=13,

        fontweight="bold"

    )


    # ========================================================
    # AXIS LABELS
    # ========================================================

    plt.xlabel(

        "Residue",

        fontsize=16,

        fontweight="bold"

    )


    plt.ylabel(

        "Membrane Penetration (Å)",

        fontsize=16,

        fontweight="bold"

    )


    # ========================================================
    # LEGEND
    # ========================================================

    legend = plt.legend(

        fontsize=13,

        frameon=False

    )


    for text in legend.get_texts():

        text.set_fontweight(
            "bold"
        )


    # ========================================================
    # AXIS STYLE
    # ========================================================

    ax = plt.gca()


    ax.spines[
        "top"
    ].set_visible(
        False
    )


    ax.spines[
        "right"
    ].set_visible(
        False
    )


    ax.spines[
        "left"
    ].set_linewidth(
        1.2
    )


    ax.spines[
        "bottom"
    ].set_linewidth(
        1.2
    )


    ax.tick_params(

        axis="both",

        width=1.2

    )


    plt.tight_layout()


    # ========================================================
    # SAVE FIGURE
    # ========================================================

    safe_region = region.replace(
        "-",
        "_"
    )


    plot_file = os.path.join(

        output_directory,

        f"barr1_barr2_{safe_region}_penetration.png"

    )


    plt.savefig(

        plot_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


    plot_files.append(
        plot_file
    )


# ============================================================
# OUTPUT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)


print("\nFrame-by-frame CSV:")
print(output_csv)


print("\nSummary CSV:")
print(summary_csv)


print("\nBead type check:")
print(bead_summary_csv)


print("\nPlots:")

for file in plot_files:

    print(file)
