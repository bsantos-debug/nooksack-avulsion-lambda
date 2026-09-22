import os
from dotenv import load_dotenv
import http.client as http_client
import logging
import argparse            # New import for command line argument parsing
import sys                 # New import required for QApplication usage
from PyQt5.QtWidgets import QApplication

from avulsionprecursors.db.config import DBConfig
from avulsionprecursors.db.sword import SWORDDatabase
from avulsionprecursors.gee.base import GEEConfig
from avulsionprecursors.pipeline.labeling import LabelingPipeline
from avulsionprecursors.gui.gui_edit_working import CrossSectionViewer

# Load environment variables from .env file
load_dotenv()

# Define processing parameters
starting_reach_id = 81170000241  # starting reach ID
start_dist = 107018              # starting distance from outlet
end_dist = 106000                # ending distance from outlet (for processing)
river_name = "Kanektok"          # river name (used in output naming)
output_dir = os.path.join(os.getcwd(), "data")  # output folder for processed files

def initialize_gee_wrapper():
    """
    Initialize Google Earth Engine using the new configuration class.
    It reads the service account and credential path from environment variables.
    """
    config = GEEConfig(
        service_account=os.getenv("GEE_SERVICE_ACCOUNT"),
        credentials_path=os.getenv("GEE_CREDENTIALS_PATH")
    )
    # GEEConfig.__post_init__ should handle initializing ee
    return config

def run_gui():
    app = QApplication(sys.argv)
    mainWin = CrossSectionViewer()
    mainWin.show()
    sys.exit(app.exec_())

def main():
    # Mode 'pipeline' execution: existing processing logic below
    # 1. Initialize Google Earth Engine (for DEM sampling later)
    initialize_gee_wrapper()
    
    # 2. Create database configuration and instantiate SWORDDatabase to check connectivity.
    db_config = DBConfig(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT")
    )
    # Instantiate SWORDDatabase; you can use this object later if needed.
    _ = SWORDDatabase(db_config)
    
    # 3. Initialize and run the labeling pipeline.
    pipeline = LabelingPipeline(
        river_name=river_name,
        output_dir=output_dir
    )
    pipeline.run(
        start_reach_id=starting_reach_id,
        start_dist=start_dist,
        end_dist=end_dist,
        skip_n=2  # For example, label every other cross-section
    )
    
    # At this stage the pipeline reaches the output point (before hydraulic processing).
    print("Processing complete up to the DEM/hydraulic stages.")

if __name__ == "__main__":
    main() 