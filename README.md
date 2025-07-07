# ⛱️ Optimal Sunshade Placement Analysis for Seoul

A data-driven analysis to identify the **optimal locations for sunshade installation** in Seoul, using **Landsat-8 satellite imagery** and **vulnerability data**.  
This project aims to reduce the impact of **urban heat islands** by prioritizing areas with high land surface temperatures (LST) and vulnerable populations.

---

## 📌 Project Overview

- **Title:** Optimal Sunshade Placement Analysis for Seoul 
- **Date:** June 2025 (Project for the 'Data Visualitzation' graduate course)

With record-breaking heatwaves becoming more frequent, protecting heat-vulnerable populations through smart infrastructure planning (e.g., sunshades) is becoming increasingly vital.  
This project proposes a method to **quantitatively evaluate sunshade necessity** and **visualize hot zones** across Seoul.

---

## 🛰️ Google Earth Engine Script

The entire LST analysis is conducted using **Google Earth Engine (GEE)**.  
You can view the script here:

🔗 **[Earth Engine Code](https://code.earthengine.google.com/a14c60e6c438f4f6880f2fc06014ed77)**

> ⚠️ **Access Note**
> - A **Google account** is required to view or run the script.
> - You must be an **approved Google Earth Engine user**.  
>   👉 [Apply here](https://signup.earthengine.google.com/)

---

## 📄 Full Presentation (PDF)

You can view the full project presentation slides (Korean) here:

📂 [`report/Presentation.pdf`](./report/Presentation.pdf)

It contains:
- Background on heatwaves in Seoul
- Data sources and preprocessing
- Definition of the **Shade Need Index (SNI)**
- K-means clustering results and map visuals
- Final selection of sunshade installation sites

---

## 🗂️ Data Sources

| Category | Source |
|----------|--------|
| **Satellite LST** | Landsat 8 Collection 2, Surface Temperature (2024-08-13) |
| **Boundary** | Seoul administrative dong (455 districts via API) |
| **Vulnerability** | Elderly heat illness data, population density |
| **Infrastructure** | Crosswalks, roads, existing sunshades, cooling shelters |

---

## 🧪 Methodology

1. **Preprocessing**
   - Remove missing LST pixels
   - Aggregate temperature stats by dong
2. **Feature Engineering**
   - Calculate temp variation: `p90 - p50`
   - Normalize using StandardScaler
3. **Shade Need Index (SNI)**  
   `SNI = 0.5 × max + 0.3 × mean + 0.2 × temp_variation`
4. **K-means Clustering (k=5)**
   - Cluster dongs by thermal profile
5. **Site Recommendation**
   - Visual overlay with road, crosswalk, and vulnerable population maps

---

## 🏙️ Key Result

Based on combined temperature data and environmental context, **면목5동** was selected as the most urgent target area for additional sunshades.  
5 high-priority installation points were identified.

---

## 🙋 Contact

**Byungseung Kong**  
Email: xncb135@korea.ac.kr  
