# **Soil Vitality Predictor (SVI)**
### Physics-informed analytics for soil biological vitality

**Repository:** [John Johnson Ogbidi – Soil Intelligence Lab](https://github.com/Johnnie7788/soil-intelligence-lab)

---

## **Overview**

The **Soil Vitality Predictor (SVI)** is a **physics-informed artificial intelligence (AI) application** designed to assess and predict soil biological vitality using geophysical, environmental, and biological observations.  
It blends **soil physics**, **geophysics**, and **machine learning** to quantify how soil health evolves in space and time, under real field conditions.

The app automatically loads multi-source data, engineers physically interpretable features, and computes a **Soil Vitality Index (SVI)** for each field and date.  
It also calibrates predictions against **Normalized Biological Index (NBI)** observations and builds a geophysical model linking **electrical resistivity (ρ)** to **volumetric soil moisture (θ)** — the key physical driver of biological activity.

---

## **System Overview**

**Field measurements → Physics-informed feature engineering → Geophysical calibration (ρ → θ model) → Biological calibration (NBI model) → Soil Vitality Index (SVI) dashboard and predictions**

Each stage builds on real measurements and interpretable parameters, ensuring that predictions remain physically realistic and biologically meaningful.

---

## **Scientific Context**

Healthy soils are complex living systems where physical, chemical, and biological processes interact dynamically.  
The Soil Vitality Predictor captures these relationships through measurable indicators grounded in science:

- **Electrical Resistivity (ρ)** — Measured in ohm-meters (Ω·m), it reflects soil structure, moisture, and salinity.  
  Low resistivity often signals wet or saline soils, while high resistivity may indicate dryness or compaction.  
  Resistivity is one of the most sensitive non-invasive proxies for subsurface soil conditions.

- **Volumetric Soil Moisture (θ)** — Expressed in cubic meters of water per cubic meter of soil (m³/m³), it defines the water available to microbes and roots.  
  Both drought and oversaturation suppress biological activity; vitality peaks between the **wilting point (WP)** and **field capacity (FC)**.

- **Soil Temperature (T)** — Temperature drives enzymatic reactions and microbial metabolism.  
  The model represents biological vitality as a bell-shaped function centered around an optimal temperature of approximately 24 °C.

- **Normalized Difference Vegetation Index (NDVI)** — A satellite-based metric of canopy greenness.  
  NDVI correlates vegetation vigor and photosynthetic activity with underlying soil function and vitality.

- **Evapotranspiration (ET)** — The combined process of soil evaporation and plant transpiration.  
  ET links soil water loss to atmospheric demand and energy balance.

- **Normalized Biological Index (NBI)** — A field-observed biological metric (such as soil respiration or enzymatic activity) used to calibrate and validate the modeled vitality.

By combining these parameters, the SVI framework connects **physical energy flows**, **geophysical soil properties**, and **biological response**, providing a scientifically coherent view of soil health.

---

## **Key Features**

- **Physics-informed feature engineering** — Derives interpretable features from resistivity, moisture, and temperature relationships instead of relying on purely statistical correlations.  

- **Geophysical soil modeling (ρ → θ)** — Builds field-specific models translating electrical resistivity into soil moisture, improving calibration for local conditions.  

- **Biological calibration (SVI → NBI)** — Uses historical, seasonal, and lagged data to align predicted vitality with measured biological indicators.  

- **Dynamic dashboard** — Presents interactive plots, time-series diagnostics, field benchmarking, and spatial maps with natural-language interpretations for non-technical users.  

- **Export-ready data products** — Outputs all engineered features, model predictions, and summaries for downstream research or advisory systems.  

---

## **Installation**

```bash
git clone https://github.com/Johnnie7788/soil-intelligence-lab.git
cd soil-intelligence-lab
pip install -r requirements.txt
streamlit run app.py
```

---

## **Usage**

1. Upload the required `.csv` datasets:  
   - `fields.csv`  
   - `weather_timeseries.csv`  
   - `soil_sensor_timeseries.csv`  
   - `geophysics.csv`  
   - `remote_sensing_ndvi.csv`  
   - `biology_observed.csv` (optional, for NBI calibration)  

2. Adjust physical parameters such as optimal resistivity, field capacity, and weighting factors directly in the sidebar.  

3. Explore the dashboard:  
   - **Response functions** show how each factor influences soil vitality.  
   - **Time-series diagnostics** reveal field trends and stress events.  
   - **Geophysical modeling** links resistivity and moisture.  
   - **NBI calibration** validates biological predictions.  

4. Export all results as `.csv` for further analysis or integration with decision-support systems.

---

## **Model Architecture**

- **Physics Module:** Defines vitality response functions for resistivity, moisture, and temperature.  

- **Feature Engineering Module:** Merges field, weather, soil, and remote sensing data into interpretable daily features.  

- **Geophysical Regression Model (ρ → θ):**  
  Uses the `HistGradientBoostingRegressor` to predict soil moisture from resistivity and environmental drivers.  

- **Biological Calibration Model (SVI → NBI):**  
  Applies grouped cross-validation (`GroupKFold`) to capture within-field consistency and cross-field generalization.

---

## **Practical Value**

- Provides **quantitative insights** into how soil responds to physical and biological stress.  
- Enables **predictive soil management** that combines remote sensing, field data, and physics-based modeling.  
- Demonstrates how **physics and geophysics** directly enhance the assessment of soil health.  
- Supports smarter, evidence-based decisions on **irrigation**, **drainage**, and **nutrient management**.  

---

## **Example Output**

The app generates:
- A mean **Soil Vitality Index (SVI)** score per field and season.  
- Response curves showing sensitivity to resistivity, moisture, and temperature.  
- Cross-validated geophysical and biological model performance reports.  
- Spatial maps visualizing soil vitality and field variability.

---

## **Author**

**John Johnson Ogbidi**  
Developer and Researcher, Soil Intelligence Lab  
[GitHub Profile](https://github.com/Johnnie7788)

---

## **License**

Released under the **MIT License** — free for research, educational, and applied use with attribution.
