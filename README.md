# GP-Transfer-Active-Learning-for-Gas-Mixture-Adsorption-Prediction-in-MOFs

### Description

This repository contains the full implementation, datasets, and environment requirements for a transfer active learning (TAL) framework for data-efficient gas mixture adsorption prediction in metal-organic frameworks (MOFs). TAL combines active learning (AL) and transfer learning (TL) to leverage pure component adsorption data for multi-component gas adsorption prediction across a range of thermodynamic conditions, simultaneously improving model generalization and reducing the data required for mixture adsorption prediction in MOFs. 

The framework is built on a Gaussian process regression (GPR) model with stacked single-target (SST) formulations for transfer from the pure-component source to the multi-component target, with uncertainty-based sampling to minimize data collection on both tasks. It was demonstrated for three mixtures (CO2/CH4, CO2/H2S, and Xe/Kr) in four MOFs (Cu-BTC, IRMOF-1, IRMOF-10, and NU-800) at pressures and temperatures up to 300 bar and 400 K. and includes tests varying the initial training data size, training data selection schemew, and AL stopping criteria

The code, adsorption datasets, and dependency specifications needed to reproduce the results are provided here

### Repository Structure

├── AF_Test                   # GP RE stopping criteria implementation for CO2/CH4 mixture across the MOFs
├── CO2_CH4                   # Contains AL and TAL implementation with LHS selection scheme and PAC stopping criteria for CO2/CH4 mixture across the MOFs
├── CO2_H2S                   # Contains AL and TAL implementation with LHS selection scheme and PAC stopping criteria for CO2/H2S mixture in Cu-BTC
├── Xe_Kr                     # Contains AL and TAL implementation with LHS selection scheme and PAC stopping criteria for Xe/Kr mixture in Cu-BTC        
├── Data_Distribution         # BI_Approach data selection scheme implementation across the MOFs and mixtures
├── Data_Size_Test            # Initial training data size test in Cu-BTC and IRMOF-1
├── Datasets                  # Adsorption data for P-X and/or P-X-T space across the MOFs and mixtures
├── Plot                      # Figure generation
├── job_script                # Main code block for TAL implementation needed across all implementation files

### TAL Framework

<img width="1798" height="1082" alt="image" src="https://github.com/user-attachments/assets/29b0ecea-9ce5-47fa-a3db-127618f92133" />

