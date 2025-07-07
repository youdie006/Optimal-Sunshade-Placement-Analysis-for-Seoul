# 🌞 Optimal Sunshade Placement Analysis for Seoul

A data-driven analysis to identify the **optimal locations for sunshade installation** in Seoul, using **Landsat-8 satellite imagery** and **vulnerability data**.  
This project aims to reduce the impact of **urban heat islands** by prioritizing areas with high land surface temperatures (LST) and vulnerable populations.

---

## 📌 Project Overview

- **Title:** 그늘막 설치 위치 최적화 분석  
- **Date:** June 2025 (for 고려대학교 데이터시각화 과제)

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
   - Aggregate temperature stats by dong (max, mean, percentiles)
2. **Feature Engineering**
   - Calculate temperature variation index: `p90 - p50`
   - Normalize with StandardScaler
3. **Define SNI (Shade Need Index)**  
   `SNI = 0.5 × max + 0.3 × mean + 0.2 × temp_variation`
4. **Clustering with K-means (k=5)**
   - Cluster similar dongs based on thermal characteristics
5. **Visualization**
   - Choropleth of SNI, t-SNE projection, radar charts

---

## 🗺️ Key Insights

- **Top-3 dongs with highest sunshade need:**
  1. 공항동
  2. 면목5동
  3. 한강로동
- **Target region selected:** **면목5동**, due to high temperature and residential exposure
- **Final Recommendation:**  
  5 additional sunshade spots identified in 면목5동 based on pedestrian traffic, shading analysis, and vulnerable population mapping.

---

## 🖼️ Visual Examples

![Shade Need Index Map](./images/sni_map_example.png)  
*Each color represents a cluster of dongs with similar LST characteristics.*

---
