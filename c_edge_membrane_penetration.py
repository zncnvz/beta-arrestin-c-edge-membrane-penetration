
# ============================================================
# IMPORTS
# ============================================================

import argparse
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# RESIDUE RANGES
# ============================================================

RESIDUE_RANGES = {
    "344-Loop": range(331, 339),
    "197-Loop": range(190, 194),
}


# ============================================================
# THREE-LETTER -> ONE-LETTER AMINO ACID CODE
# ============================================================

AA_3TO1 = {
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
    "VAL": "V",
}


# ============================================================
# MARTINI 2 SETTINGS
# ============================================================

# Phosphate bead used to define membrane leaflets
PHOSPHORUS_SELECTION = "name PO4"


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Calculate C-edge membrane penetration for "
            "beta-arrestin1 and beta-arrestin2."
        )
    )

    parser.add_argument(
        "--arr1-topology",
        required=True,
        help="Topology PDB file for beta-arrestin1.",
    )

    parser.add_argument(
        "--arr1-trajectory",
        required=True,
        help="Trajectory file for beta-arrestin1.",
    )

    parser.add_argument(
        "--arr2-topology",
        required=True,
        help="Topology PDB file for beta-arrestin2.",
    )

    parser.add_argument(
        "--arr2-trajectory",
        required=True,
        help="Trajectory file for beta-arrestin2.",
    )

    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where output files will be written.",
    )

    return parser.parse_args()


# ============================================================
# MEMBRANE PENETRATION FUNCTION
# ============================================================

def calculate_penetration(
    protein_label,
    topology,
    trajectory,
):
    """
    Calculate residue-level membrane penetration.

    Parameters
    ----------
    protein_label : str
        Name used for the protein in output tables.

    topology : str or Path
        Topology file.

    trajectory : str or Path
        Molecular dynamics trajectory.

    Returns
    -------
    pandas.DataFrame
        Frame-by-frame membrane penetration values.
    """

    print("\n" + "=" * 70)
    print(f"ANALYZING: {protein_label}")
    print("=" * 70)

    print(f"\nTopology   : {topology}")
    print(f"Trajectory : {trajectory}")

    # ========================================================
    # LOAD TRAJECTORY
    # ========================================================

    universe = mda.Universe(
        str(topology),
        str(trajectory),
    )

    print(f"\nNumber of atoms  : {universe.atoms.n_atoms}")
    print(f"Number of frames : {len(universe.trajectory)}")

    # ========================================================
    # SELECT PO4 BEADS
    # ========================================================

    po4 = universe.select_atoms(
        PHOSPHORUS_SELECTION
    )

    if po4.n_atoms == 0:
        raise ValueError(
            f"{protein_label}: No PO4 beads were found."
        )

    print(f"Number of PO4 beads: {po4.n_atoms}")

    # ========================================================
    # PREPARE RESIDUE SELECTIONS
    # ========================================================

    residue_groups = {}

    for region, resids in RESIDUE_RANGES.items():

        for resid in resids:

            # ------------------------------------------------
            # Prefer side-chain beads
            # ------------------------------------------------

            atoms = universe.select_atoms(
                f"protein and resid {resid} and not name BB"
            )

            bead_type = "sidechain"

            # ------------------------------------------------
            # If no side-chain bead exists, use BB
            # ------------------------------------------------

            if atoms.n_atoms == 0:

                atoms = universe.select_atoms(
                    f"protein and resid {resid} and name BB"
                )

                bead_type = "BB_fallback"

            # ------------------------------------------------
            # Residue not found
            # ------------------------------------------------

            if atoms.n_atoms == 0:

                print(
                    f"WARNING: {protein_label} | "
                    f"{region} | resid {resid}: "
                    "residue/bead not found."
                )

                continue

            residue_groups[(region, resid)] = {
                "atoms": atoms,
                "bead_type": bead_type,
            }

    print(
        "\nNumber of residues included:",
        len(residue_groups),
    )

    # ========================================================
    # FRAME-BY-FRAME ANALYSIS
    # ========================================================

    results = []

    for ts in universe.trajectory:

        # ----------------------------------------------------
        # PO4 Z coordinates
        #
        # MDAnalysis coordinates are in Angstrom.
        # ----------------------------------------------------

        po4_z = po4.positions[:, 2]

        # ----------------------------------------------------
        # Membrane center
        # ----------------------------------------------------

        membrane_center_z = np.mean(po4_z)

        # ----------------------------------------------------
        # Separate upper and lower leaflets
        # ----------------------------------------------------

        upper_po4 = po4_z[
            po4_z > membrane_center_z
        ]

        lower_po4 = po4_z[
            po4_z < membrane_center_z
        ]

        if (
            len(upper_po4) == 0
            or len(lower_po4) == 0
        ):
            raise ValueError(
                f"{protein_label}, frame {ts.frame}: "
                "upper/lower leaflet could not be determined."
            )

        upper_po4_z = np.mean(upper_po4)
        lower_po4_z = np.mean(lower_po4)

        # ----------------------------------------------------
        # Analyze each selected residue
        # ----------------------------------------------------

        for (
            region,
            resid,
        ), residue_info in residue_groups.items():

            atoms = residue_info["atoms"]
            bead_type = residue_info["bead_type"]

            # ------------------------------------------------
            # Residue Z coordinates
            # ------------------------------------------------

            residue_z = atoms.positions[:, 2]

            residue_center_z = np.mean(
                residue_z
            )

            # ------------------------------------------------
            # Residue located near upper leaflet
            # ------------------------------------------------

            if residue_center_z >= membrane_center_z:

                selected_leaflet = "upper"

                deepest_z = np.min(
                    residue_z
                )

                penetration_A = (
                    upper_po4_z
                    - deepest_z
                )

            # ------------------------------------------------
            # Residue located near lower leaflet
            # ------------------------------------------------

            else:

                selected_leaflet = "lower"

                deepest_z = np.max(
                    residue_z
                )

                penetration_A = (
                    deepest_z
                    - lower_po4_z
                )

            # ------------------------------------------------
            # Negative values mean no penetration
            # ------------------------------------------------

            penetration_A = max(
                0.0,
                penetration_A,
            )

            # ------------------------------------------------
            # Residue name
            # ------------------------------------------------

            resname_3 = (
                atoms.residues[0]
                .resname
                .upper()
            )

            resname_1 = AA_3TO1.get(
                resname_3,
                resname_3,
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(
                {
                    "protein": protein_label,
                    "frame": ts.frame,
                    "time_ps": ts.time,
                    "region": region,
                    "resid": resid,
                    "resname": resname_3,
                    "resname_single": resname_1,
                    "bead_type": bead_type,
                    "selected_leaflet": selected_leaflet,
                    "membrane_center_z_A": membrane_center_z,
                    "upper_PO4_z_A": upper_po4_z,
                    "lower_PO4_z_A": lower_po4_z,
                    "residue_center_z_A": residue_center_z,
                    "deepest_z_A": deepest_z,
                    "penetration_A": penetration_A,
                }
            )

    return pd.DataFrame(results)


# ============================================================
# SUMMARY
# ============================================================

def create_summary(df):
    """
    Create residue-level penetration statistics.
    """

    summary = (
        df.groupby(
            [
                "protein",
                "region",
                "resid",
                "resname",
                "resname_single",
            ]
        )["penetration_A"]
        .agg(
            mean_penetration_A="mean",
            std_penetration_A="std",
            min_penetration_A="min",
            max_penetration_A="max",
            median_penetration_A="median",
        )
        .reset_index()
    )

    return summary


# ============================================================
# BEAD TYPE SUMMARY
# ============================================================

def create_bead_summary(df):
    """
    Report whether side-chain beads or BB fallback beads
    were used for each residue.
    """

    bead_summary = (
        df[
            [
                "protein",
                "region",
                "resid",
                "resname_single",
                "bead_type",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "protein",
                "region",
                "resid",
            ]
        )
    )

    return bead_summary


# ============================================================
# PLOTTING
# ============================================================

def create_plots(
    summary,
    output_directory,
):
    """
    Generate beta-arrestin1 vs beta-arrestin2
    membrane penetration bar plots.
    """

    plot_files = []

    for region, desired_resids in RESIDUE_RANGES.items():

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

        # ====================================================
        # beta-arrestin1
        # ====================================================

        arr1_df = (
            plot_df[
                plot_df["protein"]
                == "βarr1"
            ]
            .set_index("resid")
            .reindex(residues)
        )

        # ====================================================
        # beta-arrestin2
        # ====================================================

        arr2_df = (
            plot_df[
                plot_df["protein"]
                == "βarr2"
            ]
            .set_index("resid")
            .reindex(residues)
        )

        # ====================================================
        # RESIDUE LABELS
        # ====================================================

        labels = []

        for resid in residues:

            if (
                resid in arr1_df.index
                and pd.notna(
                    arr1_df.loc[
                        resid,
                        "resname_single",
                    ]
                )
            ):

                resname = arr1_df.loc[
                    resid,
                    "resname_single",
                ]

            elif (
                resid in arr2_df.index
                and pd.notna(
                    arr2_df.loc[
                        resid,
                        "resname_single",
                    ]
                )
            ):

                resname = arr2_df.loc[
                    resid,
                    "resname_single",
                ]

            else:

                resname = "?"

            labels.append(
                f"{resname}{resid}"
            )

        # ====================================================
        # FIGURE
        # ====================================================

        plt.figure(
            figsize=(8, 6)
        )

        # beta-arrestin1
        plt.bar(
            x - width / 2,
            arr1_df[
                "mean_penetration_A"
            ].fillna(0),
            width=width,
            color="black",
            alpha=0.85,
            label="βarr1",
        )

        # beta-arrestin2
        plt.bar(
            x + width / 2,
            arr2_df[
                "mean_penetration_A"
            ].fillna(0),
            width=width,
            color="red",
            alpha=0.85,
            label="βarr2",
        )

        # ====================================================
        # ZERO LINE
        # ====================================================

        plt.axhline(
            0,
            color="gray",
            linestyle="--",
            linewidth=1.2,
        )

        # ====================================================
        # AXIS LABELS
        # ====================================================

        plt.xticks(
            x,
            labels,
            rotation=45,
            ha="right",
            fontsize=13,
            fontweight="bold",
        )

        plt.yticks(
            fontsize=13,
            fontweight="bold",
        )

        plt.xlabel(
            "Residue",
            fontsize=16,
            fontweight="bold",
        )

        plt.ylabel(
            "Membrane Penetration (Å)",
            fontsize=16,
            fontweight="bold",
        )

        # ====================================================
        # LEGEND
        # ====================================================

        legend = plt.legend(
            fontsize=13,
            frameon=False,
        )

        for text in legend.get_texts():
            text.set_fontweight("bold")

        # ====================================================
        # AXIS STYLE
        # ====================================================

        ax = plt.gca()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)

        ax.tick_params(
            axis="both",
            width=1.2,
        )

        plt.tight_layout()

        # ====================================================
        # SAVE
        # ====================================================

        safe_region = region.replace(
            "-",
            "_",
        )

        output_file = (
            output_directory
            / f"barr1_barr2_{safe_region}_penetration.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        plot_files.append(
            output_file
        )

    return plot_files


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    # ========================================================
    # INPUT PATHS
    # ========================================================

    arr1_topology = Path(
        args.arr1_topology
    )

    arr1_trajectory = Path(
        args.arr1_trajectory
    )

    arr2_topology = Path(
        args.arr2_topology
    )

    arr2_trajectory = Path(
        args.arr2_trajectory
    )

    output_directory = Path(
        args.output_dir
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # CHECK INPUT FILES
    # ========================================================

    input_files = [
        arr1_topology,
        arr1_trajectory,
        arr2_topology,
        arr2_trajectory,
    ]

    for file_path in input_files:

        if not file_path.exists():

            raise FileNotFoundError(
                f"Input file not found: {file_path}"
            )

    # ========================================================
    # SYSTEMS
    # ========================================================

    systems = {
        "βarr1": {
            "topology": arr1_topology,
            "trajectory": arr1_trajectory,
        },
        "βarr2": {
            "topology": arr2_topology,
            "trajectory": arr2_trajectory,
        },
    }

    # ========================================================
    # RUN ANALYSES
    # ========================================================

    all_data = []

    for protein_label, files in systems.items():

        df_temp = calculate_penetration(
            protein_label,
            files["topology"],
            files["trajectory"],
        )

        all_data.append(
            df_temp
        )

    # ========================================================
    # COMBINE
    # ========================================================

    df = pd.concat(
        all_data,
        ignore_index=True,
    )

    print("\n" + "=" * 70)
    print("BOTH ANALYSES COMPLETED")
    print("=" * 70)

    print("\nFirst rows:")
    print(df.head().to_string(index=False))

    # ========================================================
    # SAVE FRAME-BY-FRAME DATA
    # ========================================================

    output_csv = (
        output_directory
        / "barr1_barr2_c_edge_membrane_penetration_all_frames_A.csv"
    )

    df.to_csv(
        output_csv,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = create_summary(
        df
    )

    summary_csv = (
        output_directory
        / "barr1_barr2_c_edge_membrane_penetration_summary_A.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    print("\nSUMMARY:")
    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # BEAD TYPE CHECK
    # ========================================================

    bead_summary = create_bead_summary(
        df
    )

    bead_summary_csv = (
        output_directory
        / "barr1_barr2_c_edge_bead_type_check.csv"
    )

    bead_summary.to_csv(
        bead_summary_csv,
        index=False,
    )

    print("\nBEAD TYPE CHECK:")
    print(
        bead_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # PLOTS
    # ========================================================

    plot_files = create_plots(
        summary,
        output_directory,
    )

    # ========================================================
    # OUTPUT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print("\nFrame-by-frame CSV:")
    print(output_csv)

    print("\nSummary CSV:")
    print(summary_csv)

    print("\nBead type check:")
    print(bead_summary_csv)

    print("\nPlots:")

    for plot_file in plot_files:
        print(plot_file)


# ============================================================
# RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
