import os
import pandas as pd

def load_tess4c_files(data_dir, local_tz="America/Phoenix"):
    frames = []
    files = [f for f in sorted(os.listdir(data_dir)) if f.lower().endswith(".dat")]

    for fname in files:
        path = os.path.join(data_dir, fname)
        names = [
            "local_time", "local_time_dup", "temp_enc", "temp_sky", 
            "f1", "f1_dup", "zp1_bad", 
            "f2", "msas2", "zp2", 
            "f3", "msas3", "zp3", 
            "f4", "msas4", "zp4", "seq"
        ]
        
        df = pd.read_csv(path, sep=";", skiprows=35, header=None, names=names, on_bad_lines='skip')
        
        # 1. Parse the string into a datetime object
        df['dt'] = pd.to_datetime(df['local_time'])
        
        # 2. Localize as UTC (since the 7-hour shift proves the strings are UTC)
        df['dt'] = df['dt'].dt.tz_localize('UTC')
        
        # 3. Convert to Local Time (MST / America/Phoenix)
        df['dt'] = df['dt'].dt.tz_convert(local_tz)
        
        df = df.set_index('dt')
        frames.append(df)
        
    return pd.concat(frames).sort_index()
