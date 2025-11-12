import pandas as pd

prefix = ["bi11-3_samples", "bi11-3", "bi7-3", "bi4-6", "bi4-2", "bi2-2"]

for pre in prefix:
    df  = pd.read_csv(f"./collected_data/{pre}.csv")
    #atoms,coordinates,total_energy,forces,source
    #atoms,node_feature,global_charge,energy,force,patience,pred_energy,pred_forces,source
    # rename column 'node_feature' to 'coord', 'global_charge' to 'charge', 'energy' to 'total_energy', 'force' to 'forces'
    df = df.rename(columns={"node_feature": "coordinates", "global_charge": "charge", "energy": "total_energy", "force": "forces"})
    df.dropna(inplace=True)

    df.to_csv(f"./collected_data/{pre}.csv", index=False)
    