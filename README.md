# Final Machine Learning Project

## 📌 Project Title  
Using Customer Retail Dataset to Implement and Compare Machine Learning Models

## 🎯 Objective  
The goal of this project is to build and compare multiple machine learning models using a retail customer dataset. Through this, we learn preprocessing, visualization, model training, prediction, evaluation, and comparison of algorithms.

## 📂 Dataset Features  
- Quantity  
- UnitPrice  
- Country (target variable)

## ⚙️ Workflow  
1. Load dataset using Pandas  
2. Handle missing values  
3. Encode categorical target (`Country`) using LabelEncoder  
4. Visualize customer distribution with Matplotlib  
5. Split dataset into training and testing sets (80/20 split, stratified)  
6. Scale numeric features using StandardScaler  
7. Train models:  
   - Logistic Regression  
   - Decision Tree Classifier  
   - K‑Nearest Neighbors (KNN)  
8. Evaluate models using Accuracy and Confusion Matrix  
9. Compare performances with a bar chart  

## 📊 Results  
- Logistic Regression Accuracy: **91.4%**  
- Decision Tree Accuracy: **91.4%**  
- KNN Accuracy: **90.8%**  

Confusion matrices show class‑wise predictions, while the bar chart compares overall accuracy.

## 📈 Visualizations  
- Customer distribution by country  
- Model accuracy comparison bar chart  

## 🎓 Learning Outcomes  
- Understand the complete ML workflow  
- Apply supervised learning algorithms  
- Learn evaluation techniques (accuracy, confusion matrix)  
- Compare different ML algorithms  
- Interpret graphs and results  

## ✅ Conclusion  
This project demonstrates how different ML algorithms perform on the same dataset. Logistic Regression and Decision Tree achieved similar accuracy (~91%), while KNN was slightly lower (~90%). Students gain hands‑on experience in preprocessing, scaling, model training, and evaluation — essential steps in real‑world ML applications.

---

### 🔧 How to Run
1. Clone/download the project  
2. Place the dataset (`customer_retail csv file.csv`) in the project folder  
3. Run the script in Python 3.12+  
4. Observe printed accuracies, confusion matrices, and generated graphs  
