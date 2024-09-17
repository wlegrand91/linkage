
import pandas as pd

def read_heats_file(dh_file,uncertainty,output_file):
    """
    Read the heats file written out by the MicroCal/Origin ITC analysis
    package.

    Parameters
    ----------
    dh_file : str
        name of .dh file written out by microcal software
    output_file : str
        name of file to write out data
    uncertainty : float
        user estimate of the uncertainty on each measured heat

    Returns
    -------
    meta_data : dict
        dictionary with metadata read from the top of the file: temperature
        in Kelvin, cell and titrant concentrations in molar, and cell_volume
        in microliters
    """

    # Read data file
    with open(dh_file,'r') as f:
        lines = f.readlines()

    # Grab third line and split on ","
    meta = lines[2].split(",")

    # Parse meta data on the third line
    temperature = float(meta[0])
    stationary_cell_conc = float(meta[1])*1e-3
    titrant_syringe_conc = float(meta[2])*1e-3
    cell_volume = float(meta[3])*1e3
    
    # Split rows 6-end on "," and grab first and secon columns
    shots = []
    heats = []
    for l in lines[5:]:
        col = l.split(",")
        shots.append(float(col[0]))
        heats.append(float(col[1]))

    # Make a list of uncertainty repeated once for every observed heat
    heats_stdev = [uncertainty for i in range(len(heats))]

    # Construct dataframe with data and write out a spreadsheet
    to_df = {"injection":shots,
             "heat":heats,
             "heat_stdev":heats_stdev}
    df = pd.DataFrame(to_df)
    df.to_csv(output_file,index=False)

    # Build dictionary holding meta data
    out = {}
    out["temperature"] = temperature
    out["cell_conc"] = stationary_cell_conc
    out["titrant_conc"] = titrant_syringe_conc
    out["cell_volume"] = titrant_syringe_conc

    return out
