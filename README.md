<div align="center">
  <img height="200" alt="MHCRule" src="./figures/Logo.png/">
</div>

<div align="center">
  <em>Rule-based Model for Interpretable MHC Class I Binding Prediction.</em>
</div>

 <p align="center">
  <a href="#Introduction">Introduction</a> •
  <a href="#Requirements">Requirements</a> •
  <a href="#Usage">Usage</a> •
  <a href="#Directory Structure">Directory Structure</a> •
  <a href="#license">License</a> 
</p>

---
## Introduction

Existing predictors for MHC-I binding are predominantly deep learning-based  that are highly accurate but lack interpretability. MHCRule offers both MHC-I binding prediction and interpretable rules, enhancing trust in these predictions.

MHCRuleHydroPep is a model consisting of two submodules - MHCRulePepOnly and MHCRuleHydro. MHCRulePepOnly contains rules per allele generated using peptide sequence while MHCRuleHydro generates rules for peptide sequence encoded using a hydrophobicity scale.

<div align="center">
  <img src="./figures/MHCRule.png" alt="Project Intro" height="200" width="200"/>
  <img src="./figures/MHCRuleHydroPep.png" alt="MHCRuleHydroPep.png" height="200"/>
</div>

---
## Requirements

Python version: 3.9

See <a href="./requirements.txt"> requirements.txt</a>

---
## Usage
See <a href="./USAGE EXAMPLE.ipynb"> USAGE EXAMPLE.ipynb</a>

---
## Directory Structure

