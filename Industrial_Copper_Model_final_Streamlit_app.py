import pandas as pd
import numpy as np
import pickle
import re
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Page configuration
st.set_page_config(page_title='Industrial Copper Modeling')
st.markdown('<h1 style="text-align: center;">Industrial Copper Modeling</h1>', unsafe_allow_html=True)

# Define the values for the dropdown options
country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]
status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
# status_dict = {'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4, 'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8}
item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
# item_type_dict = {'W': 5.0, 'WI': 6.0, 'S': 3.0, 'Others': 1.0, 'PL': 2.0, 'IPL': 0.0, 'SLAWR': 4.0}
application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]
product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 640405, 640665, 164141591, 164336407, 164337175, 929423819, 1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

# Create tabs
tab1, tab2 = st.tabs(["Predict Selling Price", "Predict Status"])

with tab1:
    st.header("Predict Selling Price")
    st.write(f'<h5 style="color:red; font-size:14px;">NOTE: Enter any value - Min & Max values for reference only!</h5>',unsafe_allow_html=True)
    
    # Load the trained regression and classification models, scalers, and encoders
    with open("C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Regression_Models\\model_rf.pkl", 'rb') as file:
        loaded_model = pickle.load(file)

    with open("C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Regression_Models\\scaler.pkl", 'rb') as file:
        loaded_scaler = pickle.load(file)

    with open("C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Regression_Models\\encoder.pkl", 'rb') as file:
        loaded_encoder = pickle.load(file)

    # Collect user input for predicting selling price
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
        # Validate inputs
        pattern = "^(?:\\d+|\\d*\\.\\d+)$"
        rflag = 0
        for k in [quantity_tons_log, thickness_log, width, customer, product_ref]:
            if re.match(pattern, str(k)):
                pass
            else:
                rflag = 1
                break
        
        if rflag == 1:
            if len(str(k)) == 0:
                st.write("Please enter a valid number! Space not allowed.")
            else:
                st.write(f"You have entered an invalid number: {k}")
        else:
            if customer and country and status and item_type and application and width and product_ref and quantity_tons_log and thickness_log:
                try:
                    # Prepare the input data
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
                    # Use the fitted encoder to transform the categorical features in the new data
                    encoded_new_data = loaded_encoder.transform(new_data[['status', 'item type']])
                    encoded_new_df = pd.DataFrame(encoded_new_data, columns=loaded_encoder.get_feature_names_out(['status', 'item type']))

                    # Concatenate the encoded features with the new data
                    new_data_encoded = pd.concat([new_data.reset_index(drop=True), encoded_new_df.reset_index(drop=True)], axis=1)
                    new_data_encoded = new_data_encoded.drop(columns=['status', 'item type'])

                    # Apply the fitted scaler to all columns of the new data
                    scaled_new_data = loaded_scaler.transform(new_data_encoded)
                    new_data_scaled = pd.DataFrame(scaled_new_data, columns=new_data_encoded.columns)

                    # Ensure the order of columns matches the training data
                    final_columns = new_data_encoded.columns  # Ensure this matches the order used during training
                    new_data_final = new_data_scaled[final_columns]

                    # Make predictions using the trained model
                    y_pred_new = loaded_model.predict(new_data_final)
                    
                    # Print the predictions for the new data
                    predicted_price = np.exp(y_pred_new)[0]
                    st.success(f"Predicted Selling Price: {predicted_price}")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Please fill out all fields.")
            
with tab2:
    st.header("Predict Status")
    st.write(f'<h5 style="color:red; font-size:14px;">NOTE: Enter any value - Min & Max values for reference only!</h5>',unsafe_allow_html=True)
    
    # Load the trained Classification Model, scaler and encoder
    with open('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Classification_Models\\et_best_classifier.pkl', 'rb') as f:
        classification_model = pickle.load(f)

    with open('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Classification_Models\\standard_scaler_classifier.pkl', 'rb') as f:
        classification_scaler = pickle.load(f)

    with open('C:\\Users\\Raghu\\OneDrive\\Desktop\\Capstone_Projects\\Industrial_Copper_Modelling_Project\\Classification_Models\\one_hot_encoder_classifier.pkl', 'rb') as f:
        classification_ohe = pickle.load(f)
    
    # Collect user input for predicting status
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
        # Validate inputs
        cflag = 0
        for k in [quantity_tons_log, thickness_log, width, customer, selling_price_log]:
            if re.match(pattern, str(k)):
                pass
            else:
                cflag = 1
                break
        
        if cflag == 1:
            if len(str(k)) == 0:
                st.write("Please enter a valid number! Space not allowed.")
            else:
                st.write(f"You have entered an invalid number: {k}")
        else:
            if customer and country and item_type and application and width and product_ref and quantity_tons_log and thickness_log and selling_price_log:
                try:
                    # Prepare the input data
                    input_features = pd.DataFrame({
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
                    #  Use the fitted encoder to transform the categorical features in the new data
                    encoded_input_features = classification_ohe.transform(input_features[['item type']])
                    encoded_input_df = pd.DataFrame(encoded_input_features, columns=classification_ohe.get_feature_names_out(['item type']))

                    # Concatenate the encoded features with the new data
                    input_features_encoded = pd.concat([input_features.reset_index(drop=True), encoded_input_df.reset_index(drop=True)], axis=1)
                    input_features_encoded = input_features_encoded.drop(columns=['item type'])

                    # Apply the fitted scaler to all columns of the new data
                    scaled_input_features = classification_scaler.transform(input_features_encoded)
                    input_features_scaled = pd.DataFrame(scaled_input_features, columns=input_features_encoded.columns)

                    # Ensure the order of columns matches the training data
                    final_columns = input_features_encoded.columns  # Ensure this matches the order used during training
                    input_features_final = input_features_scaled[final_columns]

                    # Make predictions using the trained model
                    y_pred_status = classification_model.predict(input_features_final)
                    
                    if y_pred_status == 1:
                        st.write('## The Status is Won')
                    else:
                        st.write('## The Status is Lost')

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
            else:
                st.error("Please fill out all fields.")
            