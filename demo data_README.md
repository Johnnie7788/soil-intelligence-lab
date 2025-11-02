# Demo Data — *Soil Intelligence Lab*

This folder contains **synthetic (non-real)** datasets created to demonstrate and test the **Soil Vitality Predictor (SVI)** application.  
Each file mimics real-world agricultural measurements in **SI units (International System of Units)** to make the model reproducible and understandable without exposing proprietary or field data.  

## Files
| File name | Description |
|------------|--------------|
| `fields.csv` | Basic field metadata with geographic coordinates (latitude, longitude, and field IDs). |
| `weather_timeseries.csv` | Simulated daily weather data including air temperature, solar radiation, and rainfall. |
| `soil_sensor_timeseries.csv` | Modeled soil moisture and temperature readings from in-field sensors. |
| `geophysics.csv` | Synthetic apparent resistivity (ρ) data representing subsurface electrical properties. |
| `remote_sensing_ndvi.csv` | Mock remote-sensing data showing normalized difference vegetation index (NDVI). |
| `biology_observed.csv` | Sample biological activity records used for NBI (Normalized Biological Index) calibration. |

## Notes
- These datasets are **for testing and demonstration only**.    
- Users can replace them with their own site data to perform real soil vitality analysis.
