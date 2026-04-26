# Wind Resource Assessment & GIS Prediction Platform

A geospatial tool for predicting wind resource availability across U.S. regions, designed for distributed/rooftop wind energy site screening.

## Overview

Traditional wind resource assessment requires deploying meteorological towers ($50K-150K per site, 12-24 month observation period). This platform pre-screens sites using public meteorological data and spatial interpolation, dramatically reducing manual assessment effort.

## Architecture

- **Data ingestion:** NOAA ISD-Lite weather stations + ERA5 reanalysis (Open-Meteo) + USGS elevation
- **Analysis:** IDW and Kriging spatial interpolation, Weibull distribution fitting (MATLAB), wind power density calculation
- **Visualization:** Interactive Folium maps + Streamlit dashboard

## Status

🚧 Under active development. Phase tracker:

- [x] Phase 0: Foundation
- [ ] Phase 1: Data ingestion
- [ ] Phase 2: Spatial interpolation
- [ ] Phase 3: Power & suitability scoring
- [ ] Phase 4: Folium visualization
- [ ] Phase 5: Streamlit dashboard
- [ ] Phase 6: Documentation & validation

## Setup

\`\`\`bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

## License

MIT