import pandas as pd
import numpy as np
import joblib
import streamlit as st
import re
import pickle


# Page configuration
st.set_page_config(page_title='Industrial Copper Modeling')
st.markdown('<h1 style="text-align: center;">Industrial Copper Modeling</h1>', unsafe_allow_html=True)

# URL or path to your background image
background_image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTCLowDTGFq5mqUuri2XMnJ1aP8HhBjfTc4Gg&s"

# CSS to darken the background image with an overlay
st.markdown(f"""
    <style>
    .stApp {{
        position: relative;
        background-image: url('{background_image_url}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white; /* Adjust text color to be readable */
    }}
    .stApp:before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5); /* Dark overlay */
        z-index: 1;
    }}
    .stApp > div {{
        position: relative;
        z-index: 2;
    }}
    </style>
    """, unsafe_allow_html=True)

# Define the values for the dropdown options
country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407, 164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Approach","Model Comparison", "Predict Selling Price", "Predict Status", "Insights", "Recommendation"])

with tab1:
    st.header("Industry Background")
    st.markdown("""
    <div class="justified-text">
    The copper industry plays a crucial role in the global economy, providing essential raw materials for a wide range of products, including electrical wiring, plumbing, and electronics. 
    Copper mining and refining are major industrial activities, with significant operations in countries like Chile, Peru, and the United States. The industry is characterized by its complex supply chain, 
    from extraction and processing to manufacturing and recycling. As a highly conductive and versatile metal, copper is vital for sustainable energy solutions and technological advancements.
    </div>
    """, unsafe_allow_html=True)

    st.header("Problem Statement")
    st.write("""
    - The copper industry's sales and pricing data suffer from skewness and noise, impacting the accuracy of manual predictions and decision-making. Implementing a machine learning regression model with data normalization, feature scaling, and outlier detection can enhance prediction accuracy and efficiency.
    - Capturing leads in the copper industry is challenging due to inefficient evaluation processes. A lead classification model using the STATUS variable (WON for success, LOST for failure) can improve lead evaluation by focusing on relevant data points and enhancing customer conversion rates.
    """)
with tab2:
    st.header("Approach")
    st.markdown("""
    <div class="justified-text">
    <strong>1. Data Understanding</strong> 
    <ul>
        <li>Identify variable types and distributions</li>
        <li>Convert rubbish values in 'Material_Reference' starting with '00000' to null</li>
        <li>Treat reference columns as categorical variables</li>
        <li>Disregard the INDEX column</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div class="justified-text">
    <strong>2. Data Preprocessing</strong> 
    <ul>
        <li>Handle missing values with mean, median, or mode</li>
        <li>treat outliers using IQR </li>
        <li>address skewness with log transformations</li>
        <li>encode categorical variables using techniques like one-hot and label encoding</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div class="justified-text">
    <strong>3. EDA </strong>
    - Visualize outliers and skewness (before and after treatment) using Seaborn's boxplot, distplot, and violinplot.
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div class="justified-text">
    <strong>4. Feature Engineering: </strong>
    <ul>
        <li>Create new features if applicable</li>
        <li>drop highly correlated columns using SNS heatmap</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div class="justified-text">
    <strong>5. Model Building and Evaluation </strong>
    <ul>
        <li>Split the dataset into training and testing sets</li>
        <li>train and evaluate classification models using metrics</li>
        <li>optimize hyperparameters with cross-validation and grid search</li>
        <li>interpret model results based on the problem statement</li>
        <li>perform similar steps for regression modeling</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    st.markdown("""
    <div class="justified-text">
    <strong>6. Model GUI </strong>
    <ul>
        <li>Use Streamlit to create an interactive page for inputting tasks (Regression or Classification) and column values (excluding 'Selling_Price' for regression and 'Status' for classification)</li>
        <li>apply the same feature engineering and scaling steps used in model training to predict new data and display the output</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

# Sample data for regression models
regression_data = {
    "Model Name": ["Linear Regressor", "XGBoost Regressor", "ExtraTree Regressor", "DecisionTree Regressor", "RandomForest Regressor"],
    "Mean Squared Error": [2.91, 0.02, 0.02, 0.03, 0.02],
    "R2 Score": [-4.41, 0.65, 0.65, 0.52, 0.71]
}

# Sample data for classification models
classification_data = {
    "Model Name": ["DecisionTree Classifier", "ExtraTree Classifier", "XGBoost Classifier", "Logistic Regression"],
    "Accuracy": [0.90, 0.92, 0.87, 0.69],
    "Precision": [0.91, 0.92, 0.94, 0.77],
    "Recall": [0.93, 0.94, 0.89, 0.69],
    "F1 Score": [0.94, 0.95, 0.91, 0.72],
    "AUC": [0.88, 0.89, 0.85, 0.70]
}

# Create DataFrames
regression_df = pd.DataFrame(regression_data)
classification_df = pd.DataFrame(classification_data)

# Your message
message1 = "Selected RandomForest Regressor Model for predicting the Selling Price"

# HTML for green box
green_box_html1 = f"""
    <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
        <p style="color: #155724; font-size: 16px;">{message1}</p>
    </div>
    """
message2 = "Selected ExtraTree Classifier Model for predicting the Status"

# HTML for green box
green_box_html2 = f"""
    <div style="background-color: #d4edda; padding: 10px; border-radius: 5px;">
        <p style="color: #155724; font-size: 16px;">{message2}</p>
    </div>
    """

with tab3:
    st.subheader("Regression Models")
    st.write("The table below displays various algorithms used for building regression models along with their respective evaluation metrics")
    st.table(regression_df)

    # Display the green box message
    st.markdown(green_box_html1, unsafe_allow_html=True)

    st.subheader("Classification Models")
    st.write("The table below displays various algorithms used for building classification models along with their respective evaluation metrics")
    st.table(classification_df)

    # Display the green box message
    st.markdown(green_box_html2, unsafe_allow_html=True)

with tab4:
    st.header("Predict Selling Price")
    st.write('<h5 style="color:red; font-size:14px;">NOTE: Enter any value - Min & Max values for reference only!</h5>', unsafe_allow_html=True)
    
    # Load the models
    model_path = "C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Regression_Models\\model_rf_rg.joblib"
    scaler_path = "C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Regression_Models\\scaler_rg.joblib"
    encoder_path = "C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Regression_Models\\encoder_rg.joblib"

    try:
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)
        loaded_encoder = joblib.load(encoder_path)
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")

    # Collect user input
    customer = st.text_input("Customer ID (Min:12458, Max:30408185)")
    country = st.selectbox("Country:", options=country_values, key="country_price")
    status = st.selectbox("Status:", options=status_values, key="status_price")
    item_type = st.selectbox("Item Type:", options=item_type_values, key="item_type_price")
    application = st.selectbox("Application:", options=application_values, key="application_price")
    width = st.number_input("Enter Width (Min:1, Max:2990)")
    product_ref = st.selectbox("Product Reference:", options=product_ref_values, key="product_ref_price")
    quantity_tons_log = st.number_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
    thickness_log = st.number_input("Enter thickness (Min:0.18 & Max:400)")

    if st.button("Predict Selling Price"):
        pattern = "^(?:\\d+|\\d*\\.\\d+)$"
        if all(re.match(pattern, str(k)) for k in [quantity_tons_log, thickness_log, width, customer, product_ref]):
            if customer and country and status and item_type and application and width and product_ref and quantity_tons_log and thickness_log:
                try:
                    new_data = pd.DataFrame({
                        'customer': [customer],
                        'country': [country],
                        'status': [status],
                        'item type': [item_type],
                        'application': [application],
                        'width': [width],
                        'product_ref': [product_ref],
                        'quantity_tons_log': [quantity_tons_log],
                        'thickness_log': [thickness_log],
                    })
                    encoded_new_data = loaded_encoder.transform(new_data[['status', 'item type']])
                    encoded_new_df = pd.DataFrame(encoded_new_data, columns=loaded_encoder.get_feature_names_out(['status', 'item type']))
                    new_data_encoded = pd.concat([new_data.reset_index(drop=True), encoded_new_df.reset_index(drop=True)], axis=1)
                    new_data_encoded = new_data_encoded.drop(columns=['status', 'item type'])
                    scaled_new_data = loaded_scaler.transform(new_data_encoded)
                    y_pred_new = loaded_model.predict(scaled_new_data)
                    predicted_price = np.exp(y_pred_new)[0]
                    st.success(f"Predicted Selling Price: {predicted_price}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Please fill out all fields.")
        else:
            st.error("Please enter valid numeric values.")
            
with tab5:
    st.header("Predict Status")
    st.write('<h5 style="color:red; font-size:14px;">NOTE: Enter any value - Min & Max values for reference only!</h5>', unsafe_allow_html=True)
    
    # Define file paths
    model_path = "C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Classification_Models\\et_best_classifier.joblib"
    scaler_path = "C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Classification_Models\\one_hot_encoder_classifier.joblib"
    encoder_path = "C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Classification_Models\\standard_scaler_classifier.joblib"

    # Load the models and other objects
    try:
        classification_model = joblib.load(model_path)
        classification_scaler = joblib.load(scaler_path)
        classification_ohe = joblib.load(encoder_path)
    except Exception as e:
        st.error(f"An error occurred while loading the models: {e}")
    
    # Collect user input
    customer = st.text_input("Customer ID (Min:12458, Max:30408185)", key="customer_id_status")
    country = st.selectbox("Country:", options=country_values, key="status_country")
    item_type = st.selectbox("Item Type:", options=item_type_values, key="status_item_type")
    application = st.selectbox("Application:", options=application_values, key="status_application")
    width = st.number_input("Enter width (Min:1, Max:2990)", key="width_status")
    product_ref = st.selectbox("Product Reference:", options=product_ref_values, key="status_product_ref")
    quantity_tons_log = st.number_input("Enter Quantity Tons (Min:611728 & Max:1722207579)", key="quantity_tons_status")
    thickness_log = st.number_input("Enter thickness (Min:0.18 & Max:400)", key="thickness_log_status")
    selling_price_log = st.number_input("Selling Price (Min:1, Max:100001015)", key="selling_price_status")

    if st.button("Predict Status"):
        pattern = "^(?:\\d+|\\d*\\.\\d+)$"
        if all(re.match(pattern, str(k)) for k in [quantity_tons_log, thickness_log, width, selling_price_log, customer, product_ref]):
            if customer and country and item_type and application and width and product_ref and quantity_tons_log and thickness_log and selling_price_log:
                try:
                    new_data = pd.DataFrame({
                        'customer': [customer],
                        'country': [country],
                        'item type': [item_type],
                        'application': [application],
                        'width': [width],
                        'product_ref': [product_ref],
                        'quantity_tons_log': [quantity_tons_log],
                        'thickness_log': [thickness_log],
                        'selling_price_log': [selling_price_log]
                    })
                    encoded_new_data = classification_ohe.transform(new_data[['item type']])
                    encoded_new_df = pd.DataFrame(encoded_new_data, columns=classification_ohe.get_feature_names_out(['item type']))
                    new_data_encoded = pd.concat([new_data.reset_index(drop=True), encoded_new_df.reset_index(drop=True)], axis=1)
                    new_data_encoded = new_data_encoded.drop(columns=['item type'])
                    scaled_new_data = classification_scaler.transform(new_data_encoded)
                    y_pred_new = classification_model.predict(scaled_new_data)
                    # st.success(f"Predicted Status: {y_pred_new[0]}")
                    if y_pred_new == 1:
                        st.write('## The Status is Won')
                    else:
                        st.write('## The Status is Lost')

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Please fill out all fields.")
        else:
            st.error("Please enter valid numeric values.")

with tab6:
    st.subheader("Insights from Regression Model for Pricing")
    st.markdown("""
    <div class="justified-text">
        <ul>
            <li><strong>Identification of Optimal Selling Prices:</strong> Identification of optimal selling prices that maximize profitability while remaining competitive in the market.</li>
            <li><strong>Impact of Market Variables:</strong> Understanding how different market variables (e.g., demand, raw material costs, competitor pricing) influence the selling price.</li>
            <li><strong>Price Sensitivity:</strong> Insights into customer price sensitivity and how changes in price affect sales volumes.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Insights from Classification Model for Lead Evaluation")
    st.markdown("""
    <div class="justified-text">
        <ul>
            <li><strong>Lead Conversion Rates:</strong> Identification of factors that contribute to higher lead conversion rates, enabling more effective targeting of potential customers.</li>
            <li><strong>Customer Segmentation:</strong> Insights into different customer segments and their likelihood of conversion, allowing for tailored marketing strategies.</li>
            <li><strong>High-Value Leads:</strong> Detection of high-value leads with a higher probability of conversion, helping in prioritizing sales efforts.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab7:
    st.subheader("Business Recommendations")
    st.markdown("""
    <div class="justified-text">
    <ul>
        <li><strong>Optimal Pricing:</strong> Implement the regression model to set optimal selling prices, enhancing profitability.</li>
        <li><strong>Dynamic Pricing:</strong> Develop a dynamic pricing model to adjust prices based on market conditions.</li>
        <li><strong>Inventory Management:</strong> Use predictive analytics to optimize inventory levels and reduce holding costs.</li>
        <li><strong>Lead Prioritization:</strong> Utilize the classification model to prioritize leads with a higher likelihood of conversion, improving sales efficiency and customer acquisition strategies.</li>
        <li><strong>Customer Segmentation:</strong> Apply clustering algorithms to segment customers for targeted marketing.</li>
        <li><strong>Customer Lifetime Value:</strong> Predict customer lifetime value to focus on retaining high-value customers.</li>
        <li><strong>Supply Chain Efficiency:</strong> Implement models to predict and mitigate risks in the supply chain.</li>
        <li><strong>Fraud Detection:</strong> Use machine learning for detecting and preventing fraudulent activities.</li>
        <li><strong>Marketing Optimization:</strong> Utilize data analytics to optimize marketing campaigns for better results.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)