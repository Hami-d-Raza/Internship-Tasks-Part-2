# 📊 Data Science & Analytics Projects  

This repository contains **three end-to-end data projects** built using Python, Scikit-learn, Streamlit, and visualization libraries.  

1. **Term Deposit Subscription Prediction (Supervised ML – Classification)**  
2. **Customer Segmentation Using Unsupervised Learning (Clustering)**  
3. **Interactive Business Dashboard in Streamlit (Visualization & Analytics)**  

Each project demonstrates real-world **data handling, preprocessing, modeling, visualization, and business insights**.  

---

## 🔹 Project 1: Term Deposit Subscription Prediction (Bank Marketing)

### 📌 Objective
Predict whether a bank customer will subscribe to a **term deposit** using demographic and marketing campaign features.  

### ⚙️ Workflow
- **Data Handling**  
  - Robust CSV path handling (auto-detects dataset).  
  - Target (`y`) mapped from `yes/no → 1/0`.  
  - One-hot encoding for categorical variables.  

- **Models Used**  
  - Logistic Regression  
  - Random Forest Classifier  

- **Evaluation**  
  - Confusion Matrix & Classification Report  
  - F1-Score  
  - ROC Curve with AUC comparison  

- **Explainability**  
  - SHAP values for feature importance & local predictions.  

### ▶️ Run Script
```bash
python term_deposit_prediction.py
🔹 Project 2: Customer Segmentation (Mall Customers Dataset)
📌 Objective
Cluster mall customers based on spending habits and propose marketing strategies for each segment.

⚙️ Workflow
EDA: Gender distribution, Age vs Spending, Income vs Spending.

Preprocessing: Standardization of features (Annual Income, Spending Score).

Clustering:

Elbow Method for optimal clusters.

Final clustering with K=5.

Visualization:

PCA (2D linear visualization).

t-SNE (2D non-linear visualization).

Business Insights: Marketing strategies tailored for budget shoppers, luxury seekers, trendsetters, etc.

▶️ Run Script
bash
Copy code
python customer_segmentation.py
🔹 Project 3: Interactive Business Dashboard (Global Superstore Dataset)
📌 Objective
Develop an interactive Streamlit dashboard to analyze sales, profit, and performance across customer segments.

⚙️ Features
Sidebar Filters: Region, Category, Sub-Category

KPIs: Total Sales & Total Profit (auto-updated on filters)

Interactive Visualizations (Plotly):

Sales by Category

Profit by Region

Top 5 Customers by Sales

Monthly Sales & Profit Trends

Raw Data Explorer (expandable table).

▶️ Run Script
bash
Copy code
streamlit run business_dashboard.py
📂 Project Structure
bash
Copy code
📁 DS_Tasks
 ┣ 📜 term_deposit_prediction.py     # Project 1: Classification
 ┣ 📜 customer_segmentation.py       # Project 2: Clustering
 ┣ 📜 business_dashboard.py          # Project 3: Dashboard
 ┣ 📊 bank_Dataset.csv               # Dataset 1
 ┣ 📊 Mall_Customers.csv             # Dataset 2
 ┣ 📊 Global_Superstore2.csv         # Dataset 3
 ┗ 📜 README.md                      # Documentation
🛠️ Requirements
Install all dependencies before running scripts:

bash
Copy code
pip install pandas matplotlib seaborn scikit-learn shap plotly streamlit
📌 Notes
SHAP Visualizations (Project 1) → best viewed in Jupyter Notebook or Google Colab.

Streamlit Dashboard (Project 3) → run in terminal with:

bash
Copy code
streamlit run business_dashboard.py
Place datasets in the correct folder (DS_Tasks 2/).