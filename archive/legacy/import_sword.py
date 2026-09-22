#!/usr/bin/env python3
"""
Script to import SWORD v16 dataset into PostgreSQL database.
Works with GeoPackage or Shapefiles.
"""
import os
import sys
import geopandas as gpd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
def import_sword_data(file_path: str, file_type: str = 'auto'):
    """
    Import SWORD v16 dataset from GeoPackage or Shapefiles.
    
    Args:
        file_path: Path to GeoPackage file or directory containing shapefiles
        file_type: 'gpkg', 'shp', or 'auto' (detects automatically)
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
    
    # Detect file type
    if file_type == 'auto':
        if file_path.endswith('.gpkg'):
            file_type = 'gpkg'
        elif file_path.endswith('.shp'):
            file_type = 'shp'
        else:
            print("Error: Could not detect file type. Please specify 'gpkg' or 'shp'")
            return
    
    try:
        # Import reaches
        print("\n📥 Importing reaches table...")
        if file_type == 'gpkg':
            # Try common layer names
            layer_names = ['sword_reaches_v16', 'reaches', 'sword_reaches', 'sword_reaches_v16']
            reaches_gdf = None
            for layer in layer_names:
                try:
                    reaches_gdf = gpd.read_file(file_path, layer=layer)
                    print(f"   Found layer: {layer}")
                    break
                except:
                    continue
            
            if reaches_gdf is None:
                # List available layers
                import fiona
                layers = fiona.listlayers(file_path)
                print(f"   Available layers: {layers}")
                if layers:
                    print(f"   Using first layer: {layers[0]}")
                    reaches_gdf = gpd.read_file(file_path, layer=layers[0])
                else:
                    raise ValueError("No layers found in GeoPackage")
        else:  # shapefile
            if os.path.isdir(file_path):
                reaches_gdf = gpd.read_file(os.path.join(file_path, 'sword_reaches_v16.shp'))
            else:
                # Assume it's the reaches file
                reaches_gdf = gpd.read_file(file_path)
        
        reaches_gdf.to_postgis('sword_reaches_v16', engine, if_exists='replace', index=False)
        print(f"   ✅ Imported {len(reaches_gdf)} reaches")
        print(f"   Columns: {list(reaches_gdf.columns)}")
        
        # Import nodes
        print("\n📥 Importing nodes table...")
        if file_type == 'gpkg':
            layer_names = ['sword_nodes_v16', 'nodes', 'sword_nodes', 'sword_nodes_v16']
            nodes_gdf = None
            for layer in layer_names:
                try:
                    nodes_gdf = gpd.read_file(file_path, layer=layer)
                    print(f"   Found layer: {layer}")
                    break
                except:
                    continue
            
            if nodes_gdf is None:
                import fiona
                layers = fiona.listlayers(file_path)
                if len(layers) > 1:
                    print(f"   Using second layer: {layers[1]}")
                    nodes_gdf = gpd.read_file(file_path, layer=layers[1])
                else:
                    raise ValueError("Could not find nodes layer in GeoPackage")
        else:  # shapefile
            if os.path.isdir(file_path):
                nodes_gdf = gpd.read_file(os.path.join(file_path, 'sword_nodes_v16.shp'))
            else:
                # Try to find nodes file
                base_path = file_path.replace('reaches', 'nodes').replace('_reaches', '_nodes')
                nodes_gdf = gpd.read_file(base_path)
        
        nodes_gdf.to_postgis('sword_nodes_v16', engine, if_exists='replace', index=False)
        print(f"   ✅ Imported {len(nodes_gdf)} nodes")
        print(f"   Columns: {list(nodes_gdf.columns)}")
        
        print("\n✅ Import complete!")
        print(f"\n📊 Summary:")
        print(f"   - Reaches: {len(reaches_gdf)}")
        print(f"   - Nodes: {len(nodes_gdf)}")
        
    except Exception as e:
        print(f"\n❌ Error during import: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python import_sword.py <path_to_gpkg_or_shp> [gpkg|shp|auto]")
        print("\nExample:")
        print("  python import_sword.py sword_v16.gpkg")
        print("  python import_sword.py sword_v16.gpkg gpkg")
        print("  python import_sword.py ./shapefiles shp")
        sys.exit(1)
    
    file_path = sys.argv[1]
    file_type = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    import_sword_data(file_path, file_type)

