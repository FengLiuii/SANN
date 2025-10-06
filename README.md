# Supervised Learning-Based Water Quality Prediction and Ecological Risk Factor Mining across China's 12 Major River Basins
## SANN: Spatio-Temporal Aware Neural Network 

Surface water quality underpins ecosystem stability, regional security, and public health, yet capturing spatio-temporal heterogeneity from historical monitoring remains challenging. We propose a Spatio-Temporal Aware Neural Network (SANN) that couples high-order spatial structure learning with explicit temporal modeling to represent nonlinear interactions among 11 physicochemical variables across China’s 12 major river basins. Using 15,855 samples from the National Surface Water Monitoring Network, SANN is benchmarked against ten traditional, deep, and graph-based models, attaining a mean accuracy of 91.87%, an F1-score of 91.15%, and a precision of 91.32%, outperforming the state of the art water quality prediction model. Feature-importance analysis reveals distinct, time-varying regional drivers: total phosphorus dominates the eight eastern–southern basins, whereas the permanganate index prevails in the four western–northern basins. The framework clarifies spatio-temporal heterogeneity in water-quality controls and provides actionable guidance for basin-specific, time-aware pollution mitigation and ecological restoration.



## File Structure

The project includes the following main files:

* `data_processing.py`: Data preprocessing module.
* `models.py`: Defines the model architecture.
* `train_eval.py`: Script for model training and evaluation.
* `plot_roc_auc_curves.py`: Script for plotting ROC-AUC curves.
* `visualize.py`: Tool for result visualization.
* `efficiency_evaluation.py`: Module for evaluating model efficiency.
* `run_efficiency_evaluation.py`: Main script to run efficiency evaluation.
* `EFFICIENCY_EVALUATION_GUIDE.md`: Guide for efficiency evaluation.
* `water_quality_data.csv`: Dataset containing water quality indicators.

## How to Use

1. Clone the repository:

   ```bash
   git clone https://github.com/FengLiuii/SANN.git
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run model training:

   ```bash
   python train_eval.py
   ```

4. Evaluate model performance:

   ```bash
   python plot_roc_auc_curves.py
   ```
