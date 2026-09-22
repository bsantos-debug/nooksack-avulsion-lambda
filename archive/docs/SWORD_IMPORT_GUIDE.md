# SWORD v16 Dataset Import Guide

This guide will help you import the SWORD v16 dataset into your PostgreSQL database.

## Step 1: Download the SWORD v16 Dataset

1. Go to the SWORD v16 dataset Google Drive folder:
   https://drive.google.com/drive/folders/14MLBRuqqB3k0K8iAkDEd7XrhqS3_77jv

2. **Recommended: Download GeoPackage format** (.gpkg file)
   - ✅ Single file (easier to manage)
   - ✅ Modern format with better support
   - ✅ Works directly with GeoPandas
   - ✅ Easy PostgreSQL import

3. **Alternative: Shapefiles** (.shp, .shx, .dbf, .prj files)
   - ✅ Widely supported
   - ✅ Works well with GeoPandas
   - ⚠️ Multiple files to manage

4. **Not recommended: NetCDF** - More complex format, requires additional processing

## Step 2: Enable PostGIS Extension

**Important Note**: Your PostgreSQL version must match your PostGIS version. If you encounter version mismatch errors:

1. **Check your PostgreSQL version:**
   ```bash
   export PATH="/usr/local/opt/postgresql@16/bin:$PATH"
   psql --version
   ```

2. **Enable PostGIS:**
   ```bash
   psql -d sword_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
   ```

3. **If you get version mismatch errors**, you have two options:
   - **Option A**: Upgrade PostgreSQL to match PostGIS (PostgreSQL 17)
   - **Option B**: Use `ogr2ogr` (see Option A below) which will handle PostGIS setup during import

## Step 3: Import Based on File Format

### Option A: If you have GeoPackage (RECOMMENDED) ✅

This is the easiest method. Use Python/GeoPandas:

```python
import geopandas as gpd
from sqlalchemy import create_engine

# Create connection to your database
engine = create_engine('postgresql+psycopg2://BrookeMac@localhost:5432/sword_db')

# Read the GeoPackage file
# Note: Replace 'sword_v16.gpkg' with your actual filename
gpkg = gpd.read_file('sword_v16.gpkg', layer='sword_reaches_v16')  # Adjust layer name if needed

# Import reaches table
reaches_gdf = gpd.read_file('sword_v16.gpkg', layer='sword_reaches_v16')
reaches_gdf.to_postgis('sword_reaches_v16', engine, if_exists='replace', index=False)
print(f"Imported {len(reaches_gdf)} reaches")

# Import nodes table
nodes_gdf = gpd.read_file('sword_v16.gpkg', layer='sword_nodes_v16')
nodes_gdf.to_postgis('sword_nodes_v16', engine, if_exists='replace', index=False)
print(f"Imported {len(nodes_gdf)} nodes")
```

**Or using command line with ogr2ogr:**
```bash
# Import reaches
ogr2ogr -f PostgreSQL PG:"dbname=sword_db user=BrookeMac host=localhost port=5432" \
  sword_v16.gpkg -nln sword_reaches_v16 -overwrite -nlt PROMOTE_TO_MULTI

# Import nodes
ogr2ogr -f PostgreSQL PG:"dbname=sword_db user=BrookeMac host=localhost port=5432" \
  sword_v16.gpkg -nln sword_nodes_v16 -overwrite
```

### Option B: If you have Shapefiles

You'll need two shapefiles:
- `sword_reaches_v16.shp` (for reaches)
- `sword_nodes_v16.shp` (for nodes)

Use `ogr2ogr` (from GDAL) to import:

```bash
# Install GDAL tools if needed (should already be installed with PostGIS)
# Import reaches table
ogr2ogr -f PostgreSQL PG:"dbname=sword_db user=BrookeMac host=localhost port=5432" \
  sword_reaches_v16.shp -nln sword_reaches_v16 -overwrite

# Import nodes table  
ogr2ogr -f PostgreSQL PG:"dbname=sword_db user=BrookeMac host=localhost port=5432" \
  sword_nodes_v16.shp -nln sword_nodes_v16 -overwrite
```

### Option B: If you have a GeoPackage (.gpkg)

```bash
# Import reaches
ogr2ogr -f PostgreSQL PG:"dbname=sword_db user=BrookeMac host=localhost port=5432" \
  sword.gpkg sword_reaches_v16 -nln sword_reaches_v16 -overwrite

# Import nodes
ogr2ogr -f PostgreSQL PG:"dbname=sword_db user=BrookeMac host=localhost port=5432" \
  sword.gpkg sword_nodes_v16 -nln sword_nodes_v16 -overwrite
```

### Option C: If you have a PostgreSQL dump file

```bash
export PATH="/usr/local/opt/postgresql@16/bin:$PATH"
psql -d sword_db -f sword_v16_dump.sql
```

### Option D: If you have CSV files

You'll need to create tables first, then import. The tables need these columns:

**sword_reaches_v16 table:**
- `reach_id` (bigint)
- `geometry` (geometry)
- `rch_id_up` (text or text array)
- `rch_id_dn` (text or text array)
- `facc` (float)
- `river_name` (text, optional)
- `slope` (float, optional)
- Other SWORD reach attributes

**sword_nodes_v16 table:**
- `node_id` (bigint)
- `reach_id` (bigint)
- `dist_out` (float)
- `width` (float)
- `geometry` (geometry)
- `slope` (float, optional)
- Other SWORD node attributes

Then use Python/GeoPandas to import:

```python
import geopandas as gpd
from sqlalchemy import create_engine

# Create connection
engine = create_engine('postgresql+psycopg2://BrookeMac@localhost:5432/sword_db')

# Read and import reaches
reaches_gdf = gpd.read_file('sword_reaches_v16.shp')  # or .csv
reaches_gdf.to_postgis('sword_reaches_v16', engine, if_exists='replace', index=False)

# Read and import nodes
nodes_gdf = gpd.read_file('sword_nodes_v16.shp')  # or .csv
nodes_gdf.to_postgis('sword_nodes_v16', engine, if_exists='replace', index=False)
```

## Step 4: Verify the Import

Check that tables were created and have data:

```bash
export PATH="/usr/local/opt/postgresql@16/bin:$PATH"
psql -d sword_db -c "\dt"  # List all tables
psql -d sword_db -c "SELECT COUNT(*) FROM sword_reaches_v16;"
psql -d sword_db -c "SELECT COUNT(*) FROM sword_nodes_v16;"
```

## Step 5: Create Indexes (Optional but Recommended)

For better performance, create indexes:

```sql
CREATE INDEX idx_sword_reaches_reach_id ON sword_reaches_v16(reach_id);
CREATE INDEX idx_sword_reaches_geometry ON sword_reaches_v16 USING GIST(geometry);
CREATE INDEX idx_sword_nodes_reach_id ON sword_nodes_v16(reach_id);
CREATE INDEX idx_sword_nodes_node_id ON sword_nodes_v16(node_id);
CREATE INDEX idx_sword_nodes_geometry ON sword_nodes_v16 USING GIST(geometry);
```

## Troubleshooting

1. **PostGIS extension not found**: Make sure PostGIS is installed and compatible with your PostgreSQL version
2. **Permission errors**: Ensure your database user has CREATE privileges
3. **Geometry errors**: Verify that geometry columns are properly formatted (WGS84/EPSG:4326 typically)
4. **Missing columns**: The code expects specific column names. Check the table structure matches what's expected

## Need Help?

If the dataset format is different, check the actual file structure first:

```bash
# For shapefiles
ogrinfo sword_reaches_v16.shp

# For GeoPackage
ogrinfo sword.gpkg
```

