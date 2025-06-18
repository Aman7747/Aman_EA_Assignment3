# Engineering Analytics Assignment 3

**Student Name:** [Aman Desai]  
**Assignment:** Z6003 - Engineering Analytics Assignment 3  

## Assignment Description

This assignment implements two advanced neural network architectures:

1. **Physics-Informed Neural Networks (PINNs)** for cardiac activation time estimation
2. **Neural Ordinary Differential Equations (Neural ODEs)** for classification tasks

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Code

### Question 1: Physics-Informed Neural Networks

This implements PINNs for cardiac activation time estimation using the Eikonal equation.

```bash
cd src
python pinn_cardiac_activation.py
```

**What it does:**
- Generates synthetic cardiac activation data
- Creates sparse training samples using Latin Hypercube Sampling
- Trains two models:
  - Standard data-driven neural network
  - Physics-informed neural network (incorporates Eikonal equation)
- Compares performance and visualizes results

**Expected Output:**
- Training progress for both models
- RMSE comparison
- Visualization plots showing:
  - Ground truth activation times
  - Predictions from both models
  - Error maps
  - Training loss curves

### Question 2: Neural Ordinary Differential Equations

This implements Neural ODEs for classification and compares with standard neural networks.

```bash
cd src
python neural_ode_classification.py
```

**What it does:**
- Tests on multiple datasets (moons, circles, blobs, MNIST binary)
- Compares standard 1-hidden-layer NN vs Neural ODE
- Visualizes decision boundaries for 2D datasets
- Analyzes performance differences

**Expected Output:**
- Accuracy comparison for each dataset
- Decision boundary visualizations (for 2D data)
- Training curves
- Analysis of continuous vs discrete depth models

## Technical Implementation Details

### Question 1: PINNs
- **Network Architecture:** 4-layer feedforward network with Tanh activation
- **Physics Loss:** Eikonal residual: `V(x,y) * ||∇T(x,y)|| - 1 = 0`
- **Automatic Differentiation:** Used for computing gradients
- **Training:** Combined data loss + physics loss with weighting factor

### Question 2: Neural ODEs
- **ODE Solver:** Dormand-Prince (dopri5) method via torchdiffeq
- **Integration Time:** [0, 1]
- **ODE Function:** 3-layer network with Tanh activation
- **Comparison:** Standard NN vs continuous-depth Neural ODE

## Key Results and Insights

### PINNs Performance
- Physics-informed networks typically achieve lower RMSE than data-driven approaches
- Incorporation of physical constraints improves generalization with sparse data
- Eikonal equation helps regularize the solution space

### Neural ODEs Analysis
- Provides continuous-depth alternative to discrete layers
- Can model complex temporal dynamics
- Computational trade-off: more expensive but potentially more expressive
- Connection to ResNet through Euler's method discretization

## Dependencies Explained

- `torch`: PyTorch deep learning framework
- `numpy`: Numerical computing
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Dataset generation and metrics
- `scipy`: Scientific computing (Latin Hypercube Sampling)
- `torchdiffeq`: ODE solvers for Neural ODEs

## Known Issues and Assumptions

1. **Computational Requirements:** Neural ODEs are computationally intensive
2. **Convergence:** ODE solver tolerance may affect training stability
3. **Memory Usage:** Adaptive ODE solvers can use significant memory
4. **Random Seeds:** Set for reproducibility, but results may vary slightly across systems

## File Descriptions

### `pinn_cardiac_activation.py`
Complete PINN implementation including:
- Synthetic data generation
- Latin Hypercube Sampling
- Physics-informed loss computation
- Model training and evaluation
- Comprehensive visualization

### `neural_ode_classification.py`
Complete Neural ODE implementation including:
- Multiple dataset support
- Standard NN vs Neural ODE comparison
- Decision boundary visualization
- ResNet-Euler method connection demonstration

## Running Individual Components

Both scripts are self-contained and can be run independently. They will automatically:
1. Generate or load required datasets
2. Train models with progress reporting
3. Evaluate performance
4. Generate visualizations
5. Save results

## Expected Runtime

- **Question 1 (PINNs):** ~5-10 minutes on CPU, ~2-3 minutes on GPU
- **Question 2 (Neural ODEs):** ~10-15 minutes on CPU, ~3-5 minutes on GPU

## Troubleshooting

### Common Issues

1. **torchdiffeq Installation:**
   ```bash
   pip install torchdiffeq
   ```

2. **CUDA Issues:**
   - Code runs on CPU by default
   - For GPU: ensure CUDA-compatible PyTorch installation

3. **Memory Issues:**
   - Reduce batch size or network size if needed
   - Lower ODE solver tolerance for Neural ODEs

4. **Convergence Issues:**
   - Adjust learning rates
   - Modify physics loss weighting in PINNs
   - Change ODE solver method if needed

## References

- Physics-Informed Neural Networks: Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019)
- Neural ODEs: Chen, R. T., et al. (2018)
- EikonalNet: Sahli Costabal, F., et al. (2019)

## Contact

For questions about this implementation, please refer to the course materials or contact the instructor.
