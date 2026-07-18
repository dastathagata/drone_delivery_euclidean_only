import pandas as pd
import numpy as np


def generate_approximation_table(input_file,
                                 sheet_name=0,
                                 output_file="Approximation_Table.xlsx"):
    """
    Generate approximation ratio table from simulation results.

    Parameters
    ----------
    input_file : str
        Excel file containing simulation results.

    sheet_name : str or int
        Excel sheet name or index.

    output_file : str
        Output Excel filename.
    """

    # Read Excel
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    results = []

    for _, row in df.iterrows():

        k = int(row["Drone Capacity"])

        # Theoretical approximation ratio
        theoretical = 1 + (2.0 / 3.0) * (1.0 - 1.0 / k)

        # Stronger lower bound
        stronger_lb = max(row["LB"],
                          row["RLB"])

        # Observed approximation ratios
        le_ratio = row["LE"] / stronger_lb
        gmtp_ratio = row["GMTP"] / stronger_lb

        results.append({
            "Targets": int(row["Targets"]),
            "Drone Capacity": k,
            "Theoretical Approx.": round(theoretical, 3),
            "LE": round(le_ratio, 3),
            "GMTP": round(gmtp_ratio, 3)
        })

    result_df = pd.DataFrame(results)

    # Save
    result_df.to_excel(output_file, index=False)

    print(result_df)

    print(f"\nApproximation table saved to '{output_file}'")

    return result_df

generate_approximation_table(
    input_file="Drone_Final_Data.xlsx",
    output_file="Approximation_Table.xlsx"
)