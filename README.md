# Space Grave — AI-Powered LEO Collision Avoidance & Dynamic Risk Management Platform

**NASA Space Apps Challenge Project**  
B2B SaaS Decision Support Platform & Financial Risk Engine for Low Earth Orbit (LEO) Satellite Operators and Space Insurers.

---

## PROJECT SUMMARY
This project by the Space Grave team puts complex orbital dynamics, proximity scanning, and space debris tracking into action and translates them into operational decisions and financial metrics. Space Grave uses real-time Two-Line Element (TLE) data, the high-performance SGP4 Astrodynamics Engine, and NASA’s 40 J/g Energy-to-Mass Ratio (EMR) disintegration threshold to calculate a Dynamic Insurance Risk Score on a scale of 0–100, model satellite re-entry into the atmosphere, and optimize collision avoidance maneuvers to protect a $954B+ LEO space economy asset pool.

---

## Main Modules and Features

* **50 km Dynamic Security Zone and Threat Matrix:** Continuous proximity scanning is performed around active satellite fleets using automatic red alert triggers and a real-time Proximity Matrix.
* **NASA EMR 40 J/g Disintegration Standard:** Assesses disintegration risks caused by kinetic energy rather than simple geometric proximity.
* **InsurTech Policy Simulator and Dynamic Risk Score:** Evaluates physical target factors for real-time dynamic insurance underwriting and annual policy risk calculations.
* **Atmospheric Re-entry and Deorbiting Simulator:** Models structural mass, ballistic cross-sections, flight path angles, and the Sutton-Graves plasma heat distribution to simulate the end of a satellite’s operational life.
* **SGP4 Distribution in Less Than a Second:** Features a high-speed coordinate projection engine capable of scanning more than 18,500 cataloged orbital objects in real time.
* **Interactive 3D Command Center:** A dual-mode web application that provides both a public B2B portal and an interactive 3D World Radar Console.

---

## System Architecture and Data Flow

Space Grave operates on a 4-layer scalable software architecture:

1. **Data Acquisition Layer:** Retrieves live TLE orbital data sets and physical object parameters from the CelesTrak, Space-Track, and ESA DISCOS APIs.
2. **Data and Caching Layer:** Uses Redis/JSON caching to prevent exceeding external API quotas and to provide coordinate streams to the propagation engine in less than a second.
3. **Astrodynamics and Risk Calculation Engine:** Runs the Python-based SGP4 trajectory calculation, calculates 50-kilometer proximity violations, applies atmospheric reentry heat equations, and evaluates dynamic insurance risk metrics.
4. **Presentation and Alert Layer:** Built using Streamlit and PyDeck/Three.js to display the 3D Command Radar, Collision Threat Matrix, and B2B SaaS Pricing Terminal.

---

## Supported by NASA and Open Space Data

Space Grave has compiled this project by citing the following open space data sources:
* **CelesTrak & Space-Track (USSPACECOM):** Real-time TLE orbital datasets.
* **ESA DISCOS:** Mass, cross-sectional area, and RCS parameters of physical objects.
* **NASA Orbital Debris Program Office (ODPO) and OTPS:** Disintegration models, 10-year financial risk models, and 40 J/g EMR disintegration thresholds.

---

## B2B SaaS Pricing Levels

| Tier | Target Audience | Key Capabilities | Pricing Model |
| :--- | :--- | :--- | :--- |
| **Basic / Standard** | Research & Small Satellite Groups | Cataloged space debris tracking, basic maneuvering, fuel optimization | Fixed-fee monthly / annual subscription |
| **Pro / Enterprise** | Commercial Fleet Operators | Real-time anomaly detection, financial risk analysis for service interruptions | Usage-based API consumption |
| **Enterprise / InsurTech** | Space Agencies & Insurers | Cumulative financial risk modeling, 0–100 Dynamic Risk Score, custom policy issuance | Custom annual enterprise contract |

---

## Team

* **Zehra Sude Demir** — Systems Analyst & Technical Product Owner
* **Kaan Baştekin** — Team Leader & Front-End / 3D Visualization
* **Eda Tuana Bilen** — Financial Specialist
* **Nehir Koca** — Data Engineer & Presentation Manager
* **Davut Sabancı** — Data Engineer
* **Ahmet Batu Kaçar** — Backend & Astrodynamics Calculation Engine

---
*Developed for NASA Space Apps Challenge 2026.*

🌐 Canlı Web Sitesi: https://sp4c3-gr4v3.streamlit.app
