# Spectral RBD Markers

> Analysis code and scripts used for the manuscript:  
> **Cortical electrophysiological markers of REM Sleep Behaviour Disorder.**

---

## Overview

This repository contains the analysis pipeline used to generate the main results and figures for the manuscript.

It includes workflows for:

- Data preprocessing  
- Spectral analysis and aperiodic exponent calculation  
- Statistical testing of group contrasts of aperiodic exponents, band power, defooofed band power, and frequency-bin power  
- Statistical testing of associations between ACE scores and aperiodic exponents, band power, and defooofed band power  
- Figure generation  

📄 **Manuscript / preprint:** *To be added*

---

## Repository structure

- `preproc` (filtering, artefact removal, source reconstruction, parcellation)  
- `calculation_of_power_spectra` (calculation of power spectra, defooofed power spectra, band power, defooofed band power, and aperiodic exponents)  
- `aperiodic_exponent_analyses` (calculation and visualisation of GLMs contrasting aperiodic exponents between groups and investigating their association with ACE scores)  
- `defooofed_power_analyses` (calculation and visualisation of GLMs contrasting defooofed band power between groups and investigating their association with ACE scores)  
- `power_analyses` (calculation and visualisation of GLMs contrasting band power between groups)  
- `helper` (functions supporting figure generation)  

---

## Methods Summary

The analysis pipeline consists of the following major steps:

1. **Preprocessing**  
   Filtering, artefact rejection, source reconstruction, and cortical parcellation  

2. **Calculation of Power Spectra**  
   - Power spectral density (PSD) estimation using Welch's method  
   - Estimation of the aperiodic component (FOOOF) and subtraction from the PSD  

3. **Extraction of Metrics of Interest**  
   - Calculation of band power from power spectra before and after removing the aperiodic component  
   - Extraction of aperiodic exponents from FOOOF fits  

4. **Statistical Analysis & Visualization**  
   - Condition contrasts (e.g., RBD vs. HC)  
   - Associations between power spectral metrics and ACE scores  
   - Figure generation for publication  

---

## Environment & Dependencies

### Core environments

- **Preprocessing**
  - OSL-Ephys  

- **Power spectral analyses**
  - OSL-Dynamics  
  - FOOOF  

### Installation

- OSL-Ephys: __https://osl-ephys.readthedocs.io/en/latest/__  
- FOOOF: __https://fooof-tools.github.io/fooof/__  
- OSL-Dynamics: __https://osl-dynamics.readthedocs.io/en/latest/__  

We recommend using separate virtual environments for:
- Preprocessing (OSL-Ephys)  
- Power spectral analyses (OSL-Dynamics + FOOOF)  
