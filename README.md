# Direct Interval Propagation in NN Surrogates

Accompanying repository for our paper: **"Direct Interval Propagation Methods using Neural-Network Surrogates for Uncertainty Quantification in Physical Systems Surrogate Model"**.

In this work, we present a comprehensive study of **interval-based uncertainty propagation** in neural network (NN) surrogate models, including both standard **multilayer perceptrons (MLPs)** and **Deep Operator Networks (DeepONet)**. We investigate three distinct approaches:  
1. **Naïve interval propagation** through standard architectures  
2. **Bound propagation techniques** such as Interval Bound Propagation (IBP) and CROWN  
3. **Interval Neural Networks (INNs)** with interval weights  

---

## Instruction

> **Disclaimer:** The code in this repository may **not exactly replicate** the results from the paper. Differences can arise due to randomizations, parameter settings, or other implementation details.  

The repository provides implementations for three methods:  
* **Naïve approach**  
* **Interval Neural Network (INN)**  
* **Bound propagation (IBP & CROWN)**  

The **Naïve** and **INN** methods are implemented using **TensorFlow**, since the INN layers in `src/inn_layers_v2.py` were developed in this framework. While the naïve approach does not require INN layers, we implemented it in TensorFlow for consistency.  

The **bound propagation** methods are implemented using **PyTorch**, as they rely on the `Auto_LiRPA` library.  

You can try the examples by ***running the provided Jupyter notebooks***. Notebook names are self-explanatory:

| Notebook | Problem | Dataset / Setting | Method |
|----------|--------|-----------------|--------|
| `simple1dreg_aug_inn_example.ipynb` | Simple 1D regression | Interval augmentation | INN |
| `deeponet_1d_ideal_naive_example.ipynb` | 1D DeepONet | Ideal dataset | Naïve approach |  

> **Caution:** Running IBP or CROWN requires a separate PyTorch environment with `Auto_LiRPA` installed.

### Data
All dataset used in the paper are available on: https://drive.google.com/drive/folders/1BKjL8H0dDUIoBBHQQL8Fc7lIjZss4piy?usp=share_link

### Environment

#### Naïve and INN Methods
- **Python:** 3.10  
- **TensorFlow:** 2.14.0  
- **NumPy:** 1.23  
- **Scikit-learn:** 1.2.1  

#### Bound Propagation Methods
- **Python:** 3.11  
- **PyTorch:** 2.1.0  
- **NumPy:** 1.26  
- **Scikit-learn:** 1.7.1  

> **Caution:** The `inn_layers_v2.py` module is compatible **only with TensorFlow 2.14.0** and may not work correctly with newer versions.


### Final Notes

Yes, we know the repo is a bit messy. No, we’re not planning to wrap the environment in a container or anything fancy. The maintainer is trying to finish his PhD and has **zero** extra bandwidth for that level of life ambition.  

If you feel inspired (or just annoyed enough) to help clean things up, contributions are very much appreciated!
