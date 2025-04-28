import csv
import json
from mp_api.client import MPRester
import traceback
import os

# Your Materials Project API key (replace with the new 32-character key)
api_key = ''  # Replace with the new key

# Input and output CSV file paths
input_csv_path = r"D:\work\pic\test3_data.csv"
output_csv_path = r"D:\work\pic\test3_data_augmented1.csv"

# Check if input file exists
if not os.path.exists(input_csv_path):
    print(f"Error: Input file does not exist at {input_csv_path}")
    exit()

# Initialize MPRester
try:
    mpr = MPRester(api_key=api_key)
    print("MPRester initialized successfully!")
except Exception as e:
    print(f"Failed to initialize MPRester: {e}")
    traceback.print_exc()
    exit()

# Open input CSV for reading and output CSV for writing
try:
    with open(input_csv_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)  # Read the header row from the input CSV
        # Add new header columns
        new_header = header + [
            "Material ID", "Formula", "Anonymous Formula", "Is Gap Direct",
            "CBM", "VBM", "Density (g/cm^3)", "Energy per Atom (eV)",
            "Volume (cm^3)", "Band Gap (eV)", "Formation Energy (eV/atom)",
            "Fermi Energy (eV)", "Electronic Energy (eV)", "Total Energy (eV)",
            "Ionic Energy (eV)", "Equilibrium Reaction Energy per Atom (eV/atom)",
            "Elements", "Grain Boundaries", "Surface Anisotropy (J/m^2)",
            "Weighted Surface Energy",
            "Lattice a", "Lattice b", "Lattice c",
            "Lattice alpha", "Lattice beta", "Lattice gamma"
        ]
        writer.writerow(new_header)  # Write the new header to the output file

        for row in reader:
            formula_str = row[3].strip()  # Assume catalyst composition is in the fourth column (index 3)
            material_data = {}

            if formula_str:  # Ensure the catalyst composition is not empty
                try:
                    search_results = mpr.summary.search(formula=formula_str)

                    if search_results:
                        first_result = search_results[0]  # Take the first search result
                        material_id = first_result.material_id
                        material_data_summary = mpr.summary.get_data_by_id(material_id)
                        material_data_structure = mpr.materials.get_structure_by_material_id(material_id)

                        material_data = {
                            "Material ID": material_id,
                            "Formula": first_result.formula_pretty,
                            "Anonymous Formula": material_data_summary.formula_anonymous,
                            "Is Gap Direct": material_data_summary.is_gap_direct,
                            "CBM": material_data_summary.cbm,
                            "VBM": material_data_summary.vbm,
                            "Density (g/cm^3)": material_data_summary.density,
                            "Energy per Atom (eV)": material_data_summary.energy_per_atom,
                            "Volume (cm^3)": material_data_summary.volume,
                            "Band Gap (eV)": material_data_summary.band_gap,
                            "Formation Energy (eV/atom)": material_data_summary.formation_energy_per_atom,
                            "Fermi Energy (eV)": material_data_summary.efermi,
                            "Electronic Energy (eV)": material_data_summary.e_electronic,
                            "Total Energy (eV)": material_data_summary.e_total,
                            "Ionic Energy (eV)": material_data_summary.e_ionic,
                            "Equilibrium Reaction Energy per Atom (eV/atom)": material_data_summary.equilibrium_reaction_energy_per_atom,
                            "Elements": ", ".join([str(element) for element in material_data_summary.elements]),
                            "Grain Boundaries": getattr(material_data_summary, "grain_boundaries", "NA"),
                            "Surface Anisotropy (J/m^2)": material_data_summary.surface_anisotropy,
                            "Weighted Surface Energy": material_data_summary.weighted_surface_energy,
                            "Lattice a": material_data_structure.lattice.a if material_data_structure else "NA",
                            "Lattice b": material_data_structure.lattice.b if material_data_structure else "NA",
                            "Lattice c": material_data_structure.lattice.c if material_data_structure else "NA",
                            "Lattice alpha": material_data_structure.lattice.alpha if material_data_structure else "NA",
                            "Lattice beta": material_data_structure.lattice.beta if material_data_structure else "NA",
                            "Lattice gamma": material_data_structure.lattice.gamma if material_data_structure else "NA"
                        }
                        print(f"Successfully retrieved Materials Project data for {formula_str}, Material ID: {material_id}")
                    else:
                        print(f"No Materials Project data found for {formula_str}.")
                        material_data = {k: "NA" for k in new_header[len(header):]}
                except Exception as e:
                    print(f"Error processing {formula_str}: {e}")
                    material_data = {k: "Error" for k in new_header[len(header):]}
            else:
                print(f"Catalyst composition is empty, skipping Materials Project data retrieval.")
                material_data = {k: "Empty Formula" for k in new_header[len(header):]}

            # Combine original row data with Materials Project data
            output_row = row + list(material_data.values())
            writer.writerow(output_row)
except Exception as e:
    print(f"Error processing CSV files: {e}")
    traceback.print_exc()
    exit()

print(f"Data successfully exported to: {output_csv_path}")