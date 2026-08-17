# GP-Transfer-Active-Learning-for-Gas-Mixture-Adsorption-Prediction-in-MOFs

This repository contains the full implementation, datasets, and environment requirements for a transfer active learning (TAL) framework for data-efficient gas mixture adsorption prediction in metal-organic frameworks (MOFs). TAL combines active learning (AL) and transfer learning (TL) to leverage pure component adsorption data for multi-component gas adsorption prediction across a range of thermodynamic conditions, simultaneously improving model generalization and reducing the data required for mixture adsorption prediction in MOFs. 

The framework is built on a Gaussian process regression (GPR) model with stacked single-target (SST) formulations for transfer from the pure-component source to the multi-component target, with uncertainty-based sampling to minimize data collection on both tasks. It was demonstrated for three mixtures (CO2/CH4, CO2/H2S, and Xe/Kr) in four MOFs (Cu-BTC, IRMOF-1, IRMOF-10, and NU-800) at pressures and temperatures up to 300 bar and 400 K. and includes tests varying the initial training data size, training data selection schemew, and AL stopping criteria

The code, adsorption datasets, and dependency specifications needed to reproduce the results are provided here

<img width="1798" height="1082" alt="image" src="https://github.com/user-attachments/assets/29b0ecea-9ce5-47fa-a3db-127618f92133" />

