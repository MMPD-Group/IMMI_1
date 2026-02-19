# IMMI_1
This repository contains the machine learning framework developed for defect classification in Laser Powder Bed Fusion (LPBF) processed alloys. The objective of this study is to classify melt pool behavior into four defect categories: Good, Balling, Lack of Fusion (LOF), and Keyholing, using thermo-physical properties and process parameters as input features.

The study is designed around alloy-wise exclusion strategies to evaluate the generalization capability of different machine learning models. Multiple test cases are implemented where specific alloy datasets are excluded from training and later new data points for different combinations of  laser powers and scan speeds were generated and performed predictions. This enables evaluation of cross-material predictive performance.

The scripts were developed using Python 3.10. The required packages include pandas (2.2.0), numpy (1.26.0), scikit-learn (1.3.2), xgboost (1.7.6), joblib (1.3.2), shap (0.44.0), and matplotlib. The models implemented in this work include Decision Tree, Random Forest, XGBoost, Gaussian Process Classifier, Gradient Boosting Classifier, and Support Vector Machine (SVM with SVC implementation). Hyperparameter optimization is performed using GridSearchCV with cross-validation to ensure robust model selection.

Test Case-1 corresponds to the scenario where all 30 experimental data points of this work are excluded from the training dataset. The model is trained only on literature-derived data, and predictions are performed on the excluded experimental dataset. This script includes full defect classification, probability estimation for each class, and SHAP-based feature interpretation to quantify the contribution of individual thermo-physical and process parameters toward defect formation.

Test Case-2 involves excluding all AlSi10Mg data points, including both literature and experimental data, from the training dataset. The model is trained on the remaining alloys and subsequently used to predict defects for the excluded Aluminium dataset. Random Forest is primarily used for classification in this case, and both defect predictions and class probabilities are generated.

Test Case-3 follows a similar approach but excludes all SS316L data points (literature and experimental) from training. The trained model is then used to predict defect classes and probabilities for SS316L.

Test Case-4 excludes all IN718 (Inconel 718) data points, including literature and experimental data, from training. The trained classifier is used to evaluate predictive performance on the IN718 dataset, producing defect classifications and probability values.

The CSV files included in this repository correspond to the processed datasets used for each test case. The file containing “Final_ML_excluding 30 exp points” is used for Test Case-1, where the feature matrix (X) contains all thermo-physical and process parameters and the target variable (y) represents the defect class. The remaining CSV files correspond to the alloy-wise exclusion datasets used in Test Cases 2, 3, and 4.

The overall workflow implemented in each script includes data loading, preprocessing using RobustScaler, train-test splitting, hyperparameter tuning using GridSearchCV, model evaluation using accuracy, confusion matrix, and classification report, followed by defect prediction and probability estimation on the excluded dataset. In Test Case-1, SHAP analysis is additionally performed to provide interpretability and quantify feature importance across defect classes.

This repository provides a systematic framework for evaluating machine learning-based defect prediction across multiple LPBF alloys and demonstrates the generalization capability of data-driven models under alloy exclusion strategies.
