#!/bin/bash
# Script to import SWORD v16 GeoPackage files using ogr2ogr
# This avoids Python dependency issues

GPKG_DIR="/Users/BrookeMac/Library/CloudStorage/OneDrive-UW/Data/sword/gpkg"
DB_NAME="sword_db"
DB_USER="BrookeMac"
DB_HOST="localhost"
DB_PORT="5432"

echo "Importing SWORD v16 dataset into PostgreSQL..."
echo "Database: $DB_NAME on $DB_HOST:$DB_PORT"
echo ""

# Import all reach files
echo "📥 Importing reaches..."
for reach_file in "$GPKG_DIR"/*_sword_reaches_v16.gpkg; do
    if [ -f "$reach_file" ]; then
        continent=$(basename "$reach_file" | cut -d'_' -f1 | tr '[:lower:]' '[:upper:]')
        echo "   Importing $continent reaches..."
        
        # Use ogr2ogr to append to the table (will create if doesn't exist)
        # Layer name is "reaches", keep original geometry column name "geom"
        # Use POSTGIS_VERSION=2.5 to work with older PostGIS or native geometry
        ogr2ogr -f PostgreSQL \
            PG:"dbname=$DB_NAME user=$DB_USER host=$DB_HOST port=$DB_PORT" \
            "$reach_file" \
            reaches \
            -nln sword_reaches_v16 \
            -append \
            -nlt PROMOTE_TO_MULTI \
            -lco POSTGIS_VERSION=2.5 \
            -lco SPATIAL_INDEX=GIST
        
        if [ $? -eq 0 ]; then
            echo "   ✅ $continent reaches imported"
        else
            echo "   ❌ Error importing $continent reaches"
        fi
    fi
done

# Import all node files
echo ""
echo "📥 Importing nodes..."
for node_file in "$GPKG_DIR"/*_sword_nodes_v16.gpkg; do
    if [ -f "$node_file" ]; then
        continent=$(basename "$node_file" | cut -d'_' -f1 | tr '[:lower:]' '[:upper:]')
        echo "   Importing $continent nodes..."
        
        # Use ogr2ogr to append to the table
        # Layer name is "nodes", keep original geometry column name "geom"
        # Use POSTGIS_VERSION=2.5 to work with older PostGIS or native geometry
        ogr2ogr -f PostgreSQL \
            PG:"dbname=$DB_NAME user=$DB_USER host=$DB_HOST port=$DB_PORT" \
            "$node_file" \
            nodes \
            -nln sword_nodes_v16 \
            -append \
            -lco POSTGIS_VERSION=2.5 \
            -lco SPATIAL_INDEX=GIST
        
        if [ $? -eq 0 ]; then
            echo "   ✅ $continent nodes imported"
        else
            echo "   ❌ Error importing $continent nodes"
        fi
    fi
done

echo ""
echo "✅ Import complete!"
echo ""
echo "Verifying import..."
export PATH="/opt/homebrew/opt/postgresql@17/bin:$PATH"
psql -d $DB_NAME -U $DB_USER -h $DB_HOST -p $DB_PORT -c "SELECT COUNT(*) as reach_count FROM sword_reaches_v16;"
psql -d $DB_NAME -U $DB_USER -h $DB_HOST -p $DB_PORT -c "SELECT COUNT(*) as node_count FROM sword_nodes_v16;"
echo ""
echo "✅ Data imported with original structure (geometry column: 'geom')"
echo "   The code has been updated to work with the original column names."

