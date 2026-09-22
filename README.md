# River Avulsion Precursors Encoded in Alluvial Ridge Geometry

This repository contains companion tools and analysis code for the GRL paper *"River Avulsion Precursors Encoded in Alluvial Ridge Geometry"*. It provides both a processing pipeline and an interactive GUI for analyzing river cross–section data.

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/jameshgrn/avulsionprecursors
   cd avulsionprecursors
   ```

2. **Install Dependencies with Poetry**

   ```bash
   poetry install
   ```

   *Alternatively, use your preferred virtual environment mechanism to install dependencies based on the `pyproject.toml` file.*

3. **Configure Environment Variables**

   Create a `.env` file in the repository root with the following (adjust the values as needed):

   ```ini
   GEE_SERVICE_ACCOUNT=your_service_account
   GEE_CREDENTIALS_PATH=path/to/your/credentials.json
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   ```

---

## Database Setup

The project uses the SWORD v16 dataset, available [here](https://drive.google.com/drive/folders/14MLBRuqqB3k0K8iAkDEd7XrhqS3_77jv), which we have downloaded and hosted on a PostgreSQL database.

To set up your own PostgreSQL instance:

1. **Install PostgreSQL:**  
   Follow the installation instructions available at [PostgreSQL Documentation](https://www.postgresql.org/docs/current/tutorial-install.html).

2. **Create a New Database and Import the Dataset:**  
   Create a database for the SWORD v16 dataset and import the data into it. Ensure that the dataset is accessible by your application.

3. **Configure Database Credentials:**  
   Update your `.env` file with the following variables:

   ```ini
   DB_NAME=your_database_name
   DB_USER=your_database_user
   DB_PASSWORD=your_database_password
   DB_HOST=your_database_host
   DB_PORT=your_database_port
   ```

For detailed instructions on setting up your own PostgreSQL instance, refer to [this guide](https://www.codecademy.com/article/installing-and-using-postgresql-locally).

---

## How to Run

### Pipeline Mode

The pipeline processes river cross–section data (up through the DEM/hydraulic stages) and saves the results.

Run the pipeline with:

```bash
poetry run python main.py
```

> **Note:** Running `main.py` executes the full processing chain (initializing GEE, connecting to the database, and executing the labeling pipeline). Processed data are saved in the `data` directory.

### GUI Mode

The interactive GUI is provided to view and edit cross–section data.

Launch the GUI using the CLI:

```bash
poetry run python avulsionprecursors/gui/gui_edit_working.py --river VENEZ_2023
```

Replace `VENEZ_2023` with the desired river name. This will open the Cross Section Viewer with the specified river loaded.

> **Important:** The main processing pipeline and the GUI are decoupled. The pipeline (invoked via `main.py`) does not call the GUI.

---

## Code Overview

- **Pipeline Processing:**  
  Implemented in `avulsionprecursors/pipeline/labeling.py`, this module processes the DEM/hydraulic portion of the data for labeling.

- **Database Connectivity:**  
  Database configuration is handled in `avulsionprecursors/db/config.py` and `avulsionprecursors/db/sword.py`.

- **Graphical User Interface (GUI):**  
  The GUI is defined in `avulsionprecursors/gui/gui_edit_working.py` and offers interactive plotting, editing, and updating of cross–section data. It accepts a `--river` argument when launched directly.


Replace `VENEZ_2023` with the desired river name. This will open the Cross Section Viewer with the specified river loaded.

> **Important:** The main processing pipeline and the GUI are decoupled. The pipeline (invoked via `main.py`) does not call the GUI.

---

## Analysis and Plotting

In addition to the main processing pipeline and GUI, this repository now includes a comprehensive suite of analysis and plotting tools found in the `avulsionprecursors/analysis` directory. These modules include:

- **Regression Analysis:**  
  Files such as `LA_LC_regression.py` and `lengthscale_regression.py` are used for performing regression analyses on river morphology metrics.

- **Spatial and Statistical Analysis:**  
  The `LISA.py` file provides tools for Local Indicators of Spatial Association (LISA) analysis, while `based.py` implements the BASED (Boost Assissted Stream Estimator for Depth) methodology for analyzing river channels.

- **Variogram Modeling:**  
  Files like `variogram.py` and `variogram_implementation.py` allow you to model and analyze the spatial structure of the river data through variograms.

- **Proximity Analysis:**  
  The `proximity_analysis.py` file provides tools for analyzing the proximity of avulsion precursors to large $\Lambda$ values.

- **Plotting and Visualization:**  
  Modules including `plot_lambda.py`, `plot_mapview.py`, and `wavelengths.py` offer tools to generate plots of avulsion potential, map views of river data, and wavelength analyses, respectively. Additionally, `lengthscale_histogram.py` provides histogram visualizations of key length scales within the dataset.

These analysis tools enable detailed investigation and visualization of avulsion potential ($\Lambda$), facilitating a deeper understanding of the processes controlling river avulsion.

---

## Code Overview

- **Pipeline Processing:**  
  Implemented in `avulsionprecursors/pipeline/labeling.py`, this module processes the DEM/hydraulic portion of the data for labeling.

- **Database Connectivity:**  
  Database configuration is handled in `avulsionprecursors/db/config.py` and `avulsionprecursors/db/sword.py`.

- **Graphical User Interface (GUI):**  
  The GUI is defined in `avulsionprecursors/gui/gui_edit_working.py` and `avulsionprecursors/gui/labeler.py` for interactive plotting and editing of cross–section data.

- **Analysis and Plotting:**  
  All analysis and visualization modules are contained within the `avulsionprecursors/analysis` directory. These include regression, spatial statistics, variogram analysis, and assorted plotting scripts that complement the processing pipeline and GUI workflows.


---

## Citation

If you use this code in your research, please cite:

*River Avulsion Precursors Encoded in Alluvial Ridge Geometry*, GRL.

---

## License

This project is licensed under MIT.

---


