# BASED Analyzer Requirements

## What the Original BASED Code Requires

### 1. Input: SwordReach Object

The `BASEDAnalyzer.analyze_reach(reach: SwordReach)` method expects:

#### Nodes Must Have:
- `node_id`: Unique identifier
- `reach_id`: Reach identifier
- `dist_out`: Distance from outlet (meters)
- `width`: Channel width (meters)
- `slope`: Channel slope (unitless)
- `elevation`: Node elevation (meters, optional)
- `cross_section`: LineString geometry (REQUIRED - nodes without this are skipped)

### 2. DataFrame Columns Required After `_nodes_to_dataframe`

The initial DataFrame from `_nodes_to_dataframe` contains:
- `node_id`
- `reach_id`
- `dist_out`
- `width`
- `slope`
- `elevation`

### 3. Additional Columns Required Before Processing

#### REQUIRED: Discharge
- **Column name**: `discharge_value`
- **Type**: float (m³/s)
- **When needed**: Before `_calculate_discharge()` is called
- **Error if missing**: `ValueError("Discharge value missing from data")`
- **Usage**: 
  - Corrected using inverse power law: `corrected_discharge = (discharge_value / a) ^ (1/b)`
  - Used with width and slope to predict depth: `depth = XGBoost(width, slope, corrected_discharge)`

#### REQUIRED: Cross-Section Labels
The DataFrame must have these columns before `_calculate_ridge_parameters()` is called:

**Minimum (single side):**
- `channel_dist_along`: Distance along cross-section to channel center (meters)
- `channel_elevation`: Elevation at channel center (meters)
- `ridge1_dist_along`: Distance along cross-section to ridge 1 (meters)
- `ridge1_elevation`: Elevation at ridge 1 (meters)
- `floodplain1_dist_along`: Distance along cross-section to floodplain 1 (meters)
- `floodplain1_elevation`: Elevation at floodplain 1 (meters)

**Optional (both sides):**
- `ridge2_dist_along`: Distance along cross-section to ridge 2 (meters)
- `ridge2_elevation`: Elevation at ridge 2 (meters)
- `floodplain2_dist_along`: Distance along cross-section to floodplain 2 (meters)
- `floodplain2_elevation`: Elevation at floodplain 2 (meters)

### 4. Processing Steps

The `analyze_reach()` method:
1. Creates DataFrame from nodes (`_nodes_to_dataframe`)
2. Clips slope to minimum
3. **Requires `discharge_value` column** → calculates `corrected_discharge`
4. Predicts depth → creates `XGB_depth` column
5. **Requires label columns** → calculates ridge parameters
6. Calculates gamma → `gamma_mean`
7. Calculates superelevation → `superelevation_mean`
8. Calculates lambda → `lambda = gamma_mean * superelevation_mean`

### 5. Summary

**What must be in the DataFrame BEFORE calling BASED:**

1. **From nodes (automatic):**
   - `node_id`, `reach_id`, `dist_out`, `width`, `slope`, `elevation`

2. **Must be added (REQUIRED):**
   - `discharge_value`: One value per node (can be same for all nodes)
   - `channel_dist_along`, `channel_elevation`
   - `ridge1_dist_along`, `ridge1_elevation`
   - `floodplain1_dist_along`, `floodplain1_elevation`
   - `ridge2_dist_along`, `ridge2_elevation` (optional)
   - `floodplain2_dist_along`, `floodplain2_elevation` (optional)

### 6. Key Points

- **Width and Slope**: Must be in nodes (from shapefiles)
  - Extracted from node attributes: `node.width`, `node.slope`
  - Read from shapefile columns when building reach
  - Used with discharge to predict depth
  - **REQUIRED** - cannot be skipped

- **Discharge**: Must be provided as `discharge_value` column in DataFrame
  - Can be same value for all nodes (typical)
  - Can vary per node (if discharge changes along reach)
  - **NOT** in shapefiles - must be added separately
  - Required for depth prediction
  
- **Depth Prediction**: Uses ALL THREE inputs:
  - `depth = XGBoost(width, slope, corrected_discharge)`
  - All three are required - cannot predict depth without width, slope, AND discharge
  
- **Labels**: Must be provided as columns in DataFrame
  - Extracted from labeled cross-sections
  - Must have at least one side (ridge1, floodplain1)
  - Second side (ridge2, floodplain2) is optional

- **Cross-sections**: Nodes must have `cross_section` LineString
  - Nodes without cross-sections are skipped
  - Cross-section geometry defines the line along which distances are measured

## Example Usage

```python
# 1. Build reach with nodes
reach = build_reach_from_shapefiles(...)

# 2. Create initial DataFrame (from BASED)
analyzer = BASEDAnalyzer(model_path, params_path)
df = analyzer._nodes_to_dataframe(reach)

# 3. ADD REQUIRED COLUMNS
# Add discharge (required)
df['discharge_value'] = 100.0  # or per-node values

# Add labels (required)
df['channel_dist_along'] = ...  # from labeled cross-sections
df['channel_elevation'] = ...
df['ridge1_dist_along'] = ...
# ... etc

# 4. Run analysis
df['slope'] = df['slope'].clip(lower=analyzer.min_slope)
df = analyzer._calculate_discharge(df)  # Requires discharge_value
df = analyzer._predict_depth(df)  # Requires corrected_discharge
df = analyzer._calculate_ridge_parameters(df)  # Requires label columns
df = analyzer._calculate_gamma(df)
df = analyzer._calculate_superelevation(df)
df['lambda'] = df['gamma_mean'] * df['superelevation_mean']
```
