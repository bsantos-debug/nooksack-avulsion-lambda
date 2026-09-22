#!/usr/bin/env python3
"""
Script to import all SWORD v16 continent GeoPackage files into PostgreSQL database.
Combines all continents into unified sword_reaches_v16 and sword_nodes_v16 tables.
"""
import os
import sys
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

def import_all_sword_data(gpkg_dir: str):
    """
    Import all SWORD v16 continent files into PostgreSQL.
    
    Args:
        gpkg_dir: Directory containing the GeoPackage files
    """
    # Get database connection from environment
    db_name = os.getenv('DB_NAME', 'sword_db')
    db_user = os.getenv('DB_USER', 'BrookeMac')
    db_password = os.getenv('DB_PASSWORD', '')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    
    # Build connection string
    if db_password:
        conn_str = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    else:
        conn_str = f'postgresql+psycopg2://{db_user}@{db_host}:{db_port}/{db_name}'
    
    print(f"Connecting to database: {db_name} on {db_host}:{db_port}")
    engine = create_engine(conn_str)
    
    gpkg_path = Path(gpkg_dir)
    
    # Find all reach and node files
    reach_files = sorted(gpkg_path.glob('*_sword_reaches_v16.gpkg'))
    node_files = sorted(gpkg_path.glob('*_sword_nodes_v16.gpkg'))
    
    print(f"\n📁 Found {len(reach_files)} reach files and {len(node_files)} node files")
    
    if not reach_files or not node_files:
        print("❌ Error: Could not find SWORD GeoPackage files")
        print(f"   Looking in: {gpkg_dir}")
        return
    
    # Import all reaches
    print("\n📥 Importing reaches from all continents...")
    all_reaches = []
    
    for i, reach_file in enumerate(reach_files, 1):
        continent = reach_file.stem.split('_')[0].upper()
        print(f"   [{i}/{len(reach_files)}] Loading {continent} reaches... ", end='', flush=True)
        try:
            gdf = gpd.read_file(str(reach_file))
            # Try to find the geometry column
            if 'geometry' not in gdf.columns:
                # Try common geometry column names
                geom_cols = [c for c in gdf.columns if gdf[c].dtype == 'geometry']
                if geom_cols:
                    gdf = gdf.set_geometry(geom_cols[0])
                    gdf = gdf.rename(columns={geom_cols[0]: 'geometry'})
            
            all_reaches.append(gdf)
            print(f"✅ ({len(gdf):,} reaches)")
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if all_reaches:
        print(f"\n   Combining {len(all_reaches)} reach datasets...")
        combined_reaches = gpd.GeoDataFrame(pd.concat(all_reaches, ignore_index=True))
        print(f"   Total: {len(combined_reaches):,} reaches")
        
        # Ensure CRS is set
        if combined_reaches.crs is None:
            print("   ⚠️  No CRS found, setting to EPSG:4326 (WGS84)")
            combined_reaches.set_crs('EPSG:4326', inplace=True)
        
        print(f"   Writing to database...")
        combined_reaches.to_postgis('sword_reaches_v16', engine, if_exists='replace', index=False)
        print(f"   ✅ Successfully imported {len(combined_reaches):,} reaches")
        print(f"   Columns: {list(combined_reaches.columns)}")
    else:
        print("   ❌ No reaches imported")
    
    # Import all nodes
    print("\n📥 Importing nodes from all continents...")
    all_nodes = []
    
    for i, node_file in enumerate(node_files, 1):
        continent = node_file.stem.split('_')[0].upper()
        print(f"   [{i}/{len(node_files)}] Loading {continent} nodes... ", end='', flush=True)
        try:
            gdf = gpd.read_file(str(node_file))
            # Try to find the geometry column
            if 'geometry' not in gdf.columns:
                geom_cols = [c for c in gdf.columns if gdf[c].dtype == 'geometry']
                if geom_cols:
                    gdf = gdf.set_geometry(geom_cols[0])
                    gdf = gdf.rename(columns={geom_cols[0]: 'geometry'})
            
            all_nodes.append(gdf)
            print(f"✅ ({len(gdf):,} nodes)")
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    if all_nodes:
        print(f"\n   Combining {len(all_nodes)} node datasets...")
        combined_nodes = gpd.GeoDataFrame(pd.concat(all_nodes, ignore_index=True))
        print(f"   Total: {len(combined_nodes):,} nodes")
        
        # Ensure CRS is set
        if combined_nodes.crs is None:
            print("   ⚠️  No CRS found, setting to EPSG:4326 (WGS84)")
            combined_nodes.set_crs('EPSG:4326', inplace=True)
        
        print(f"   Writing to database...")
        combined_nodes.to_postgis('sword_nodes_v16', engine, if_exists='replace', index=False)
        print(f"   ✅ Successfully imported {len(combined_nodes):,} nodes")
        print(f"   Columns: {list(combined_nodes.columns)}")
    else:
        print("   ❌ No nodes imported")
    
    print("\n" + "="*60)
    print("✅ Import complete!")
    print(f"\n📊 Summary:")
    if all_reaches:
        print(f"   - Total Reaches: {len(combined_reaches):,}")
    if all_nodes:
        print(f"   - Total Nodes: {len(combined_nodes):,}")
    print("\n💡 You can now use the SWORD database with your pipeline!")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        default_path = "/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/sword/gpkg"
        print(f"Usage: python import_sword_all.py <gpkg_directory>")
        print(f"\nDefault path: {default_path}")
        print(f"\nUsing default path...")
        gpkg_dir = default_path
    else:
        gpkg_dir = sys.argv[1]
    
    if not os.path.isdir(gpkg_dir):
        print(f"❌ Error: Directory not found: {gpkg_dir}")
        sys.exit(1)
    
    import_all_sword_data(gpkg_dir)

