# Beta-Arrestin C-Edge Membrane Penetration Analysis

This repository contains a Python script for residue-level membrane penetration analysis of selected C-edge regions of β-arrestin1 (βarr1) and β-arrestin2 (βarr2) using coarse-grained molecular dynamics trajectories.

The analysis is designed for Martini 2 membrane systems and uses phosphate (`PO4`) beads to define the membrane leaflets.

## Analysis Overview

The script evaluates membrane penetration of two selected C-edge regions:

* **344-Loop:** residues 331–338
* **197-Loop:** residues 190–193

The same residue numbering is used for both βarr1 and βarr2.

For each residue and each trajectory frame, the script determines its position relative to the membrane and calculates its penetration depth with respect to the corresponding membrane leaflet.

## Membrane Reference

Membrane phosphate beads are selected using:

```text
name PO4
```

The membrane center is calculated from the mean Z-coordinate of all PO4 beads.

PO4 beads are then separated into:

* Upper leaflet
* Lower leaflet

The mean Z-coordinate of the phosphate beads in each leaflet is used as the membrane reference position.

## Residue Representation

For each selected residue, side-chain beads are used whenever available.

If a residue does not contain a side-chain bead, the Martini backbone bead (`BB`) is used as a fallback.

The selected bead type is reported in the output files.

## Penetration Calculation

For residues associated with the upper leaflet, the deepest residue bead is defined as the bead with the lowest Z-coordinate.

Penetration is calculated as:

```text
Penetration = Upper leaflet PO4 Z - Deepest residue Z
```

For residues associated with the lower leaflet, the deepest residue bead is defined as the bead with the highest Z-coordinate.

Penetration is calculated as:

```text
Penetration = Deepest residue Z - Lower leaflet PO4 Z
```

Negative penetration values are set to zero.

Therefore:

```text
0 Å   = no membrane penetration
>0 Å  = membrane penetration
```

All distances are reported in Ångström (Å).

## Requirements

The script requires:

* Python 3
* MDAnalysis
* NumPy
* Pandas
* Matplotlib

Install the required packages using:

```bash
pip install MDAnalysis numpy pandas matplotlib
```

## Input Files

The script requires one topology file and one trajectory file for each β-arrestin system.

Example:

```python
arr1_topology = "basal_arr2_membrane.pdb"
arr1_trajectory = "basal_arr2_membrane.xtc"

arr2_topology = "basal_arr3_membrane.pdb"
arr2_trajectory = "basal_arr3_membrane.xtc"
```

Users should modify these file names or paths in the `USER SETTINGS` section of the script according to their own system.

In the original dataset used for this analysis, files named `arr2` correspond to βarr1 and files named `arr3` correspond to βarr2.

## Output Directory

The output directory can be defined in the script:

```python
output_directory = "output"
```

The directory is created automatically if it does not already exist.

## Output Files

The script generates the following outputs.

### Frame-by-Frame Penetration Data

```text
barr1_barr2_c_edge_membrane_penetration_all_frames_A.csv
```

This file contains penetration measurements for every analyzed residue and every trajectory frame.

Reported information includes:

* Protein
* Frame number
* Simulation time
* C-edge region
* Residue number
* Residue name
* Bead type
* Selected membrane leaflet
* Membrane center
* Upper and lower PO4 positions
* Residue Z-position
* Deepest residue bead position
* Membrane penetration

### Residue-Level Summary

```text
barr1_barr2_c_edge_membrane_penetration_summary_A.csv
```

For each residue, the following statistics are calculated:

* Mean penetration
* Standard deviation
* Minimum penetration
* Maximum penetration
* Median penetration

### Bead-Type Check

```text
barr1_barr2_c_edge_bead_type_check.csv
```

This file reports whether each residue was analyzed using:

* Side-chain beads
* BB fallback

### Plots

Separate bar plots are generated for the two analyzed C-edge regions:

```text
barr1_barr2_344_Loop_penetration.png
barr1_barr2_197_Loop_penetration.png
```

The plots compare the mean membrane penetration of βarr1 and βarr2 for each residue.

## Running the Script

After defining the input files in the `USER SETTINGS` section, run:

```bash
python c_edge_membrane_penetration.py
```

## Script

The main analysis script is:

```text
c_edge_membrane_penetration.py
```

## Notes

This analysis was developed for coarse-grained Martini 2 membrane simulations.

The membrane penetration measurement is based on the Z-coordinate of Martini phosphate beads and therefore assumes that the membrane is primarily oriented in the XY plane with the membrane normal along the Z-axis.

