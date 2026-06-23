import pandas as pd
from pathlib import Path

xls_path = Path("data/reference/1-s2.0-S1872497312001810-mmc7.xls")
xl = pd.ExcelFile(xls_path)
print("Sheet names:", xl.sheet_names)

for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    print(f"\nSheet '{sheet}' shape: {df.shape}")
    print("Columns:", list(df.columns)[:10])
    # Let's save it to csv in scratch for easy viewing
    csv_path = Path(f"scratch/sheet_{sheet}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved sheet '{sheet}' to {csv_path}")
