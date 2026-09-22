#!/usr/bin/env python3
"""
Simple SWORD import using GeoPandas - works around PostGIS issues
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import geopandas as gpd
    from sqlalchemy import create_engine
except ImportError as e:
    print(f"Error: Missing required packages. Please install: {e}")
    sys.exit(1)

GPKG_DIR = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/sword/gpkg"

# Get database connection
db_name = os.getenv('DB_NAME', 'sword_db')
db_user = os.getenv('DB_USER', 'BrookeMac')
db_password = os.getenv('DB_PASSWORD', '')
db_host = os.getenv('DB_HOST', 'localhost')
db_port = os.getenv('DB_PORT', '5432')

if db_password:
    conn_str = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
else:
    conn_str = f'postgresql+psycopg2://{db_user}@{db_host}:{db_port}/{db_name}'

print(f"Connecting to: {db_name}@{db_host}:{db_port}")
engine = create_engine(conn_str)

# Import reaches
print("\n📥 Importing reaches...")
reach_files = sorted(Path(GPKG_DIR).glob('*_sword_reaches_v16.gpkg'))
all_reaches = []

for i, reach_file in enumerate(reach_files, 1):
    continent = reach_file.stem.split('_')[0].upper()
    print(f"   [{i}/{len(reach_files)}] {continent}... ", end='', flush=True)
    try:
        gdf = gpd.read_file(str(reach_file), layer='reaches')
        all_reaches.append(gdf)
        print(f"✅ ({len(gdf):,})")
    except Exception as e:
        print(f"❌ Error: {e}")

if all_reaches:
    print(f"\n   Combining {len(all_reaches)} datasets...")
    import pandas as pd
    combined = gpd.GeoDataFrame(pd.concat(all_reaches, ignore_index=True))
    print(f"   Total: {len(combined):,} reaches")
    print(f"   Writing to database...")
    combined.to_postgis('sword_reaches_v16', engine, if_exists='replace', index=False)
    print(f"   ✅ Reaches imported!")

# Import nodes  
print("\n📥 Importing nodes...")
node_files = sorted(Path(GPKG_DIR).glob('*_sword_nodes_v16.gpkg'))
all_nodes = []

for i, node_file in enumerate(node_files, 1):
    continent = node_file.stem.split('_')[0].upper()
    print(f"   [{i}/{len(node_files)}] {continent}... ", end='', flush=True)
    try:
        gdf = gpd.read_file(str(node_file), layer='nodes')
        all_nodes.append(gdf)
        print(f"✅ ({len(gdf):,})")
    except Exception as e:
        print(f"❌ Error: {e}")

if all_nodes:
    print(f"\n   Combining {len(all_nodes)} datasets...")
    import pandas as pd
    combined = gpd.GeoDataFrame(pd.concat(all_nodes, ignore_index=True))
    print(f"   Total: {len(combined):,} nodes")
    print(f"   Writing to database...")
    combined.to_postgis('sword_nodes_v16', engine, if_exists='replace', index=False)
    print(f"   ✅ Nodes imported!")

print("\n✅ Import complete!")

