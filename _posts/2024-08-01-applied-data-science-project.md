
<img width="928" height="337" alt="Screenshot 2025-08-31 at 7 12 30 PM" src="https://github.com/user-attachments/assets/127bc482-7537-4c86-975a-fe5b58091a62" /><img width="928" height="555" alt="Screenshot 2025-08-31 at 7 12 52 PM" src="https://github.com/user-attachments/assets/b71b9eac-c8a2-46ac-af85-b2c8aeff2aef" />

## **Yaamanni Kasavan** | ITD 214

## Project Background
This project focuses on Kapital Fashion, a global fashion brand aiming to strengthen its market position through data-driven decision-making. As the fashion industry becomes increasingly competitive, understanding customer behavior, optimizing pricing models, and enhancing engagement strategies are critical to sustaining growth and loyalty.

The primary goal of this project is to optimize Kapital Fashion’s product offerings, pricing strategies, and customer engagement in order to maximize sales and strengthen brand loyalty in the global market.

To achieve this goal, we have identified four key business objectives:

1️⃣ Analyze the relationship between product condition (new, gently used, used, etc.) and price to determine its impact on pricing tiers.

2️⃣ Forecast customer loyalty using review ratings and purchase frequency.

3️⃣ Segment customers to design targeted discount strategies that drive future sales.

4️⃣ Leverage text analytics from online fashion discussions to inform and refine Kapital’s global product, pricing, and engagement strategies.

Through these objectives, the project **combines quantitative analysis (pricing, segmentation, forecasting) with qualitative insights (text analytics)** to deliver a comprehensive, data-backed strategy for Kapital Fashion’s long-term success.

### Data Preparation
**Dataset Details**
- Source: Kaggle
- Size: 3,900 rows × 19 columns
- Key focus areas: Consumer shopping trends, transactional details, demographics, and behavior
- Column Names (19 total):
<img width="688" height="363" alt="Screenshot 2025-08-31 at 5 04 01 PM" src="https://github.com/user-attachments/assets/ae629fb9-3e8c-4db5-a67a-fa4d1e6d166f" />

**Data Quality & Structure**
- No missing values in any of the 19 columns.
- Customer Age: ranges from 18 to 70 (mean ~44).
- Review Rating: ranges from 2.5 to 5.0 (mean ~3.75).
- Purchase Amount (USD): ranges from 20 to 100 (mean ~59.76).
- Previous Purchases: ranges from 1 to 50 (mean ~25.35).
- Categorical variables (e.g., Gender, Location, Subscription Status, Payment Method, Shipping Type, Discount Applied, Promo Code Used, etc.) have limited unique values, making them suitable for encoding.

**Data Quality Assessment:**
- Completeness: No missing values (0 missing across all columns).
- Accuracy: No obvious outliers; Review Rating fits 1-5 scale.
- Duplicates: Customer ID uniqueness to be verified.

**Data Exploration:**
- Age distribution leans toward middle-aged customers.
- Purchase Amount skewed, with most between $39 and $81.
- Review Rating indicates positive feedback trend
 <img width="507" height="161" alt="Screenshot 2025-08-31 at 5 05 43 PM" src="https://github.com/user-attachments/assets/78163adf-ae58-4bf4-a99a-5407fe5ade09" />
 
<img width="817" height="141" alt="Screenshot 2025-08-31 at 5 09 51 PM" src="https://github.com/user-attachments/assets/d0279257-2d36-4816-8e46-ec0899cfccc0" />

1. The dataset was first reduced from 19 columns to five key attributes—Customer ID, Review Rating, Age, Previous Purchases, and Frequency of Purchases—to maintain analytical focus.
2. Missing values were assessed and removed, ensuring that the dataset was complete.
3. Duplicate rows were eliminated, preserving unique Customer IDs.
4. Data types were standardized, with Customer ID stored as a string, Review Rating as a double, and both Age and Previous Purchases as integers, ensuring consistency across variables.
5. Outliers were detected and capped, specifically for Age (18–100), Review Rating (1–5), and Previous Purchases (≤100), resulting in a dataset free of anomalies.
<img width="915" height="500" alt="Screenshot 2025-08-31 at 5 07 02 PM" src="https://github.com/user-attachments/assets/3d6b92a1-c43e-43c9-a11b-440eb8cd46a2" />


### Modelling

**Modelling Technique**
- Technique: Binary classification to predict Loyal (Weekly/Bi-Weekly/Fortnightly/Monthly, ≥12 purchases/year) vs. Non-Loyal (Every 3 Months/Quarterly/Annually, <12 purchases/year).
**Models:**
- Logistic Regression: Linear, interpretable, baseline for binary classification.
- SVM (RBF kernel): Captures non-linear patterns, effective with balanced data (post-SMOTE).
- KNN: Instance-based, leverages local patterns, sensitive to scaling.
**Rationale:**
Binary classification simplifies the 7-class problem, aligning with retention by segmenting customers into actionable groups.
Logistic Regression is straightforward and interpretable for business use.
SVM and KNN add non-linear capabilities to capture potential patterns, despite weak correlations.
SMOTE addresses slight imbalance (~56% Loyal, 44% Non-Loyal).
Excludes multi-class and regression to keep it straightforward, as they underperformed (accuracy ~0.15, R² ~ -0.03).

**Test Design**
- Data: 3900 rows (Customer ID, Age, Review Rating, Previous Purchases, Frequency of Purchases). 
- Preprocessing:
- Target: Loyalty_Binary (1 = Frequent, ≥12 purchases/year; 0 = Infrequent). Mapping: Weekly=52, Bi-Weekly=26, Fortnightly=24, Monthly=12, Every 3 Months/Quarterly=4, Annually=1.
- Features: Age, Review Rating, Previous Purchases (no interactions for simplicity).
- EDA: Correlation matrix (Age, Review Rating, Previous Purchases vs. Frequency_Numeric) and binary target distribution.
- Split: 80% train (3120 rows), 20% test (780 rows), random_state=42.
- Standardization: StandardScaler for all models (Logistic Regression, SVM, KNN require scaled features).
- Balancing: SMOTE to balance Loyal/Non-Loyal classes in training.
- Tuning: 5-fold cross-validation with GridSearchCV, scoring='accuracy'.
- Evaluation: Test set for final metrics; visualizations (heatmap, confusion matrices, scatter plot) for insight.
- Baseline: Random guessing accuracy ~0.50 (balanced classes).

**Build Model**
- Class Balancing with SMOTE:
SMOTE (Synthetic Minority Over-sampling Technique) is applied to the training data to balance the Loyal and Non-Loyal classes by generating synthetic samples for the minority class.
This prevents model bias toward the majority class (~56% Loyal) and improves F1-scores for imbalanced data.
Model Training and Tuning: Each model is tuned using GridSearchCV with 5-fold cross-validation on the SMOTE-balanced training data, optimizing for 'accuracy'.
- Logistic Regression:
Parameters tuned: C (regularization strength: [0.1, 1, 10]), solver ('lbfgs' for optimization).
Model is fit on balanced data, predicting probabilities/logits for binary classes.
- SVM:
Parameters tuned: C ([0.1, 1, 10]), kernel ('rbf' for non-linear decision boundaries).
Set probability=True to enable probability predictions for visualizations.
Fit on balanced data, using RBF kernel to capture potential non-linear patterns in features.
- KNN:
Parameters tuned: n_neighbors ([3, 5]), weights ('distance' for weighted voting).
Fit on balanced data, classifying based on nearest neighbors in feature space.

## Model Assessment 

**Determine Criteria**

**- Primary Metric: Accuracy** | The proportion of correct predictions (True Loyal + True Non-Loyal) divided by total predictions. It measures overall model correctness.
Why Chosen: Simple and business-friendly, directly answers "How often is the model right in identifying loyal customers?" For binary classification, it's intuitive for retention decisions (e.g., if accuracy is ~0.52, ~52% of targeted customers are truly Loyal).
Threshold: Aim for >0.50 (better than random guessing); lower indicates weak features.
Expansion: In imbalanced datasets (~56% Loyal), accuracy can be misleading if biased toward the majority class, so it's paired with F1-score for balance.

**- Secondary Metric: Macro F1-Score** | Harmonic mean of precision (correct Loyal predictions among predicted Loyal) and recall (correct Loyal predictions among actual Loyal), averaged across classes without weighting by class size.
Why Chosen: Balances precision (avoiding false Loyal predictions, which wastes retention resources) and recall (capturing most actual Loyal customers for targeting). Macro averaging treats Loyal and Non-Loyal equally, addressing slight imbalance.
Threshold: Aim for >0.50; values ~0.50-0.51 (as seen) indicate modest performance but highlight class-specific weaknesses (e.g., lower recall for Non-Loyal).
Expansion: Breaks down into precision/recall per class in the classification report, allowing assessment of business risks (e.g., low Loyal recall means missing retention opportunities).

**- Visual and Diagnostic Metrics: Confusion Matrix** | A 2x2 table showing true positives (correct Loyal), true negatives (correct Non-Loyal), false positives (Non-Loyal predicted as Loyal), and false negatives (Loyal predicted as Non-Loyal).
Why Chosen: Provides actionable insights beyond numbers, e.g., false positives waste marketing budgets, false negatives miss loyal customers. Visual heatmaps make it easy to spot imbalances.
Threshold: Minimize false negatives for retention focus (prioritize capturing Loyal customers).
Expansion: Used to derive business costs (e.g., cost of false positive = unnecessary reward; cost of false negative = lost repeat purchase). In results, matrices show ~ balanced errors (~175-260 correct per class, estimated).

**- General Criteria: Cross-Validation During Tuning** | 5-fold cross-validation accuracy in GridSearchCV to ensure the model generalizes to unseen data.
Why Chosen: Prevents overfitting on training data, validating reliability for real-world forecasting.
Threshold: Close to test accuracy (~0.48-0.50 in prior tuning).
Expansion: Averages performance across folds, flagging variance (e.g., if CV accuracy << test accuracy, model may overfit).

**- Interpretability and Visualization Criteria** | Qualitative assessment via visualizations (confusion matrices, scatter plots of predictions vs. Review Rating).
Why Chosen: Business stakeholders need visual insights to trust and act on predictions (e.g., scatter plot shows if high Review Ratings correlate with Loyal probabilities).
Threshold: Clear patterns in plots (e.g., higher probabilities for actual Loyal); weak patterns indicate feature limitations.
Expansion: Confusion matrices quantify errors visually; scatter plot expands assessment by linking predictions to key features like Review Rating, helping refine strategies (e.g., target high-rating Loyal predictions).

✴️ **ASSESS MODEL 1 :**
**LOGISTIC REGRESSION:**
- Accuracy: ~0.4961
- Macro F1: ~0.50 (balanced, precision/recall ~0.48-0.50).
- Confusion Matrix: ~175/359 Non-Loyal correct, ~212/421 Loyal correct (estimated from accuracy).
- Assessment: Near random guessing, limited by linear assumptions and weak features.
  
  <img width="295" height="212" alt="Screenshot 2025-08-31 at 7 04 56 PM" src="https://github.com/user-attachments/assets/7cd29221-2d7a-40b0-a74b-d33ca4599a34" />


✴️ **ASSESS MODEL 2 :**
**SVM (RBF):**
- Accuracy: ~0.50
- Macro F1: ~0.49
- Confusion Matrix: ~146/359 Non-Loyal correct, ~244/421 Loyal correct.
- Assessment: Best performer, captures some non-linear patterns
  
<img width="304" height="212" alt="Screenshot 2025-08-31 at 7 05 17 PM" src="https://github.com/user-attachments/assets/d537f14a-f6b8-4f5e-b055-bbc7380d62f6" />


✴️ **ASSESS MODEL 3:**
**KNN:**
- Accuracy: ~0.4923
- Macro F1: ~0.49
- Confusion Matrix: ~180/359 Non-Loyal correct, ~204/421 Loyal correct.
- Assessment: Comparable to Logistic Regression, sensitive to scaling.
  
<img width="281" height="205" alt="Screenshot 2025-08-31 at 7 05 35 PM" src="https://github.com/user-attachments/assets/67c45d5b-0d15-44df-a299-a5a94b648d21" />



### Evaluation
**Model Evaluation and Final Selection**

**Logistic Regression**
- Strengths: Simple and interpretable; coefficients show the impact of features; computationally efficient.
- Weaknesses: Limited to linear relationships; struggled with weak correlations (~0.00); achieved accuracy of ~0.49, close to random guessing.
- Fit: Moderate, but constrained by weak feature relevance.

**Support Vector Machine (SVM, RBF Kernel)**
- Strengths: Captures non-linear patterns; achieved slightly higher accuracy than logistic regression; improved recall for Loyal customers (~0.60).
- Weaknesses: Computationally intensive; still limited by weak predictors, with performance only marginally better than random.
- Fit: Best among the three, as it maximizes accuracy and recall within the dataset’s constraints.

**K-Nearest Neighbors (KNN)**
- Strengths: Conceptually simple; capable of identifying local data patterns.
- Weaknesses: Highly sensitive to feature scaling; accuracy remained ~0.49 with no clear performance advantage.
- Fit: Comparable to logistic regression, offering no additional benefit.

**Final Model Choice: SVM (RBF)**
- Expected performance: Accuracy ~0.50, F1 ~0.56.
- Selected as the most suitable option because it marginally outperforms the alternatives, leverages non-linear relationships, and offers better recall for Loyal customers—supporting retention strategies despite overall limitations.

**Objectives Achieved (Partially):**
- Models successfully predicted binary loyalty status (Loyal vs. Non-Loyal) using available features, enabling basic customer segmentation.
- Confusion matrices and related visualizations provided insights into classification accuracy and feature limitations.
- Practical application: Loyal customers can be targeted with rewards or retention strategies.

**Limitations:**
- Low overall accuracy limits reliability for precise loyalty forecasting.
- Weak feature correlations indicate the chosen predictors do not fully capture true loyalty drivers, reducing strategic effectiveness.

<img width="320" height="242" alt="Screenshot 2025-08-31 at 6 08 36 PM" src="https://github.com/user-attachments/assets/2e4b580f-6f15-4521-9e72-dfa765a5a353" />
<img width="281" height="229" alt="Screenshot 2025-08-31 at 6 08 59 PM" src="https://github.com/user-attachments/assets/3a7b2b6d-7d59-478d-85e5-5130a08c095e" />
<img width="289" height="226" alt="Screenshot 2025-08-31 at 6 09 25 PM" src="https://github.com/user-attachments/assets/d6560b9e-0307-4ecc-bacb-05ea73a8106e" />


## Recommendation and Analysis
<img width="819" height="427" alt="Screenshot 2025-08-31 at 7 29 14 PM" src="https://github.com/user-attachments/assets/53b91dfc-f190-42ee-9cb4-ef4e2156af86" />


**Business Issues**

The modeling process revealed several critical business issues that impacted the ability to effectively forecast customer loyalty (proxied by Frequency of Purchases as Loyal vs. Non-Loyal). These issues stem from data limitations, methodological challenges, and broader organizational gaps in understanding loyalty drivers. Below is an expanded explanation, incorporating insights from the code's EDA and results (accuracy near random guessing):

**1. Weak Predictors and Low Feature-Target Correlations:**
The selected predictors (Age, Review Rating, Previous Purchases) have negligible correlations with Frequency_Numeric, as shown in the correlation matrix and heatmap visualization. This indicates the features provide little discriminatory power, leading to models performing barely better than random guessing.
In retail, loyalty is often driven by factors beyond demographics or ratings, such as emotional experiences, product quality, or personalized service. The absence of these results in models that fail to capture true loyalty patterns, wasting resources on ineffective predictions. For example, if Review Rating doesn't strongly influence frequency (correlation ~ -0.002), the model can't reliably identify high-rating customers as loyal.

**2. Insufficient Initial Exploratory Data Analysis (EDA):**
The code includes EDA (correlations, distributions), but this should have been conducted earlier to validate predictors before modeling. The binary distribution (~56% Loyal, 44% Non-Loyal) shows slight imbalance, and visualizations like the heatmap highlight no strong relationships, yet modeling proceeded without pivoting to new features.
This oversight led to inefficient resource use (e.g., tuning models on weak data). In business terms, it reflects a gap in data-driven decision-making, assuming Age or Review Rating drives loyalty without validation could misdirect retention efforts, such as targeting older customers (Age correlation ~ -0.001) who may not be more loyal.

**3. Class Overlap and Imbalance in Feature Space:**
Even after SMOTE balancing, the confusion matrices (visualized for each model) show balanced errors but high misclassifications, indicating overlap in feature distributions between Loyal and Non-Loyal customers.
This suggests the features don't separate classes well. e.g., similar Review Ratings across groups (scatter plot shows scattered probabilities with no clear pattern). Business-wise, this risks inefficient marketing: false positives waste budgets on Non-Loyal customers, while false negatives miss retention opportunities for actual Loyal ones.

**4. Missing Key Data and Loyalty Drivers:**
The dataset lacks critical retail loyalty drivers like purchase amount, recency, product type, or engagement metrics, limiting model performance to low accuracy.
Retail loyalty is multifaceted studies show it's driven by service quality, product quality, and brand image, or emotional factors like self-esteem and fun shopping experiences. Without these, the model can't forecast accurately, leading to suboptimal retention strategies that ignore personalized service or value-added offerings.

## Recommendations: 

To address the issues and improve loyalty forecasting for retention strategies, I recommend a phased approach: short-term fixes using the current model, medium-term enhancements, and long-term data strategies. These leverage the code's insights and external research on loyalty drivers, incorporating high-quality products, emotional experiences, and personalized service.

**Deploy and Utilize the Best Model (SVM)**:
Implement the SVM model to segment customers into Loyal vs. Non-Loyal, using predictions for initial targeting.
Use the scatter plot visualization to prioritize high-probability Loyal customers with Review Rating >4.0. For example, send automated rewards (e.g., discounts) to predicted Loyal customers, potentially boosting repeat purchases by 10-20% based on loyalty program benchmarks. Monitor via dashboards integrating confusion matrix insights to track misprediction costs.

**Launch Targeted Retention Strategies:**
Loyal Customers: Offer personalized incentives like loyalty points or exclusive access, focusing on emotional drivers (e.g., fun shopping experiences).
Non-Loyal Customers: Use re-engagement tactics, such as proactive live chat or value-added services (e.g., free shipping).

**Enhance Data Collection and Integration:**
Collect new predictors like purchase amount, recency, product category, or engagement metrics (e.g., app usage, live chat interactions).
Use customer surveys or CRM data to capture emotional drivers (e.g., self-esteem from shopping) and technology-enabled personalization (e.g., AI recommendations). Integrate external benchmarks (e.g., from web searches on loyalty programs) to add features like membership status, potentially boosting accuracy to ~0.60-0.70.

**Long-Term Organizational Changes:**
Foster a data-driven culture: Prioritize EDA in future projects to avoid weak predictors.
Invest in loyalty programs inspired by successful retail examples (e.g., tiered rewards), focusing on innovation and customer service to drive emotional loyalty. Track ROI: Aim for 10-15% revenue uplift from better retention, monitoring shifts in loyalty drivers


## DID NOT MEET BUSINESS OBJECTIVE ❌ - Relying solely on review ratings and purchase frequency to forecast customer loyalty does not effectively meet business objectives, as the model’s accuracy is only marginally better than random guessing.

## AI Ethics
<img width="500" height="246" alt="Screenshot 2025-08-31 at 7 33 02 PM" src="https://github.com/user-attachments/assets/2a5cf50e-566c-4a15-9921-2b6baaa97483" />


**1. Privacy**
**Issue:** The dataset includes sensitive attributes (Customer ID, Age, Review Rating, Previous Purchases), posing privacy risks if mishandled.

**Context:** Customer ID is a unique identifier, and Age is protected under regulations like GDPR/CCPA. Combining these with purchase behavior (Previous Purchases, Frequency) risks re-identification if linked with external data (e.g., public X profiles). The code lacks explicit security measures (e.g., encryption).
Risks:
**Data Breaches:** Insecure storage could expose customer details, leading to legal penalties or reputational harm.
Re-identification: Linking Customer ID with external sources could reveal identities, especially for unique Age/Purchase combinations.
Intrusive Profiling: Using Age and purchase data for loyalty predictions may feel invasive without clear consent.

**Mitigations:**
Anonymization: Remove or hash Customer ID before modeling. Bin Age (e.g., 18-25, 26-35) to reduce specificity.
Data Security: Use encrypted databases with role-based access controls. Implement secure pipelines (e.g., HTTPS for data transfers).
Consent: Obtain explicit customer consent for using Age and purchase data in loyalty predictions, disclosing purposes per GDPR/CCPA.
Minimal Data Use: Exclude non-essential fields like Customer ID from modeling (as done in the code).


**2. Fairness**
**Issue:** Models may introduce bias, particularly by Age, leading to unfair treatment across customer groups.

**Context**: Age (18-70) has a weak correlation with loyalty (~ -0.001). The binary target (~56% Loyal, 44% Non-Loyal) uses SMOTE for balance, but fairness across Age groups wasn’t tested. The scatter plot (SVM probabilities vs. Review Rating) shows no clear Age patterns, but bias risks remain. Risks:Age Bias: Younger or older customers may be misclassified as Non-Loyal, affecting reward allocation (e.g., younger customers with fewer purchases excluded).Feature Bias: Review Rating may reflect subjective experiences unevenly, amplifying demographic biases.Retention Disparity: Unequal reward distribution could reduce loyalty or cause dissatisfaction among certain groups.

**Mitigations:**
Fairness Audits: Evaluate predictions by Age groups (e.g., <30, 30-50, >50) using metrics like demographic parity or equal opportunity.
Feature Evaluation: Remove Age if it adds bias without predictive value. Test alternatives like purchase recency.
Balanced Targeting: Ensure equitable reward distribution across demographics, monitored via A/B testing.
Fairness-Aware Models: Use techniques like adversarial training if biases emerge with new data.



**3. Accuracy**
**Issue:** Low model accuracy (~0.51-0.53 for SVM, ~0.49 for Logistic Regression, ~0.50 for KNN) undermines reliability for retention decisions.

**Context:** Weak feature correlations (~0.00-0.01) result in near-random performance (baseline ~0.50). Confusion matrices show ~40-50% misclassifications, and the scatter plot reveals no clear prediction patterns. This limits effective loyalty forecasting.
Risks:
**Misclassification Costs:** False positives waste marketing budgets; false negatives miss loyal customers, reducing revenue.
Customer Experience: Inaccurate targeting (e.g., excluding Loyal customers from rewards) may increase churn.
Business Decisions: Relying on ~0.51 accuracy risks poor ROI and misinformed strategies.

**Mitigations:**
Improve Data: Collect predictors like purchase amount, recency, or product type to boost accuracy (>0.60).
Fallback Rules: Use rules (e.g., Review Rating >4.0, Previous Purchases >30) if accuracy remains low.
Transparent Reporting: Communicate ~0.51 accuracy to stakeholders, highlighting limitations.
Continuous Monitoring: Track errors post-deployment using confusion matrix insights to refine strategies.



**4. Accountability**
**Issue:** Lack of clear ownership for model predictions risks unaddressed errors or customer harm.

**Context:** The code automates predictions but doesn’t define who monitors outcomes or handles complaints (e.g., misclassified customers missing rewards). Retention actions (e.g., rewards) have real-world impacts, requiring accountability.
Risks:
**Customer Harm:** False negatives exclude loyal customers from benefits, with no redress process.
Organizational Risk: Undefined roles may delay error correction, risking financial or legal issues.
Stakeholder Trust: Lack of accountability reduces confidence in data science initiatives.

**Mitigations:**
Assign Ownership: Designate a data science lead to monitor performance and address feedback, using dashboards for error tracking.
Feedback Mechanism: Implement an appeal process for reward exclusions, linked to Customer ID (with privacy safeguards).
Audit Trails: Log predictions and decisions for traceability and audits.
Regular Reviews: Conduct quarterly reviews of model outcomes and A/B test results to adjust protocols.



**5. Transparency**
**Issue:** Limited communication of model limitations (~0.51 accuracy, weak predictors) to stakeholders or customers risks distrust and misinformed decisions.

**Context:** The code provides metrics (accuracy, F1, confusion matrices) and visualizations (heatmap, scatter plot), but there’s no mechanism to explain these to non-technical audiences. Low accuracy and weak correlations need clear disclosure.
Risks:
Stakeholder Misunderstanding: Business leaders may overestimate model reliability, leading to flawed strategies.
Customer Distrust: Customers may feel unfairly targeted without understanding prediction logic.
Regulatory Non-Compliance: Lack of explainability risks violating GDPR’s requirement for automated decision-making transparency.

**Mitigations:**
Explainable Outputs: Create simplified reports summarizing accuracy (~0.51), limitations (weak features), and visuals (e.g., confusion matrix).
Customer Communication: Disclose data-driven targeting to customers with opt-out options, ensuring GDPR/CCPA compliance.
Model Interpretability: Use Logistic Regression coefficients (if viable) to explain feature impacts (e.g., “Higher Review Rating slightly increases Loyal probability”).
Documentation: Maintain detailed records of code, data, and limitations (e.g., “Accuracy limited by missing predictors”), shared with stakeholders.

**Conclusion**
The project faces ethical challenges: privacy risks from Customer ID/Age, fairness concerns from potential Age bias, low accuracy (~0.51) impacting reliability, accountability gaps in error handling, and transparency issues in communicating limitations. Mitigations include anonymization, fairness audits, better data, clear ownership, and simplified reporting, aligning with retail loyalty ethics (consent, equity, trust).

## Source Codes and Datasets
Kaggle File:
https://www.kaggle.com/datasets/bhadramohit/customer-shopping-latest-trends-dataset
