import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import qrcode
from PIL import Image
import io


server_ip = "192.168.1.255"
port = "8800"
url = "https://bakeabetterbake.streamlit.app/"

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_L,
    box_size=10,
    border=4,
)
qr = qrcode.QRCode()
qr.add_data(url)
qr.make(fit=True)

buffer = io.BytesIO()
qr.make_image(fill_color="black", back_color="white").save(buffer, format="PNG")
buffer.seek(0)




baked_goods_dataset = pd.read_csv('categorized_baked_goods_dataset_final.csv', encoding='latin1')



st.title("Bake a Better Bake: A Baked Goods Ratio Optimizer")
st.write(f"To share with other please use this URL: {url}")


baked_goods_images = {
    "Brownie": "https://www.biggerbolderbaking.com/wp-content/uploads/2021/03/Best-ever-brownies-thumbnail-scaled.jpg",
    "Gluten Free Brownie": "https://mintandmallowkitchen.com/wp-content/uploads/2021/03/Fudgy-Gluten-Free-Brownies-Horizontal-2.jpg",  
    "Vegan Brownie": "https://static01.nyt.com/images/2019/02/11/dining/cd-vegan-brownies-with-tahini-and-halvah/merlin_150546963_0dd87ddb-109d-4d15-8af0-c1aed1081759-jumbo.jpg",
    "Vanilla Cupcake": "https://www.allrecipes.com/thmb/i9KCEbxUGQ1Sa4F7Gts7SGBOpoM=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/157877-vanilla-cupcakes-ddmfs-4X3-0397-59653731be1d4769969698e427d7f5bc.jpg",
    "Vegan Vanilla Cupcake": "https://images.immediate.co.uk/production/volatile/sites/30/2022/01/Vegan-Vanilla-Cupcakes-f5c67f8.jpg?resize=768,574",
    "Gluten Free Vanilla Cupcake": "https://ichef.bbci.co.uk/food/ic/food_16x9_832/recipes/gluten-free_vanilla_98272_16x9.jpg",
    "Chocolate Cupcake": "https://hips.hearstapps.com/hmg-prod/images/190220-kalhua-cupcakes-horizontal-1553011882.png",
    "Vegan Chocolate Cupcake": "https://static01.nyt.com/images/2024/09/26/multimedia/Vegan-Chocolate-Cupcakesrex-fgpv/Vegan-Chocolate-Cupcakesrex-fgpv-threeByTwoMediumAt2X.jpg",
    "Gluten Free Chocolate Cupcake": "https://static01.nyt.com/images/2024/09/26/multimedia/Vegan-Chocolate-Cupcakesrex-fgpv/Vegan-Chocolate-Cupcakesrex-fgpv-threeByTwoMediumAt2X.jpg",
    "Chocolate Chip Cookies": "https://static01.nyt.com/images/2022/02/12/dining/JT-Chocolate-Chip-Cookies/JT-Chocolate-Chip-Cookies-threeByTwoMediumAt2X.jpg",
    "Vegan Chocolate Chip Cookie": "https://anotherveganfoodblog.com/wp-content/uploads/2023/12/final-aerial-horizontal-1-1024x683.webp",
    "Gluten Free Chocolate Chip Cookie": "https://confessionsofagroceryaddict.com/wp-content/uploads/2022/12/horizontal-hero-of-Oat-flour-chocolate-chip-cookies.jpg"
}

baked_goods_options = [
    "Brownie", "Gluten Free Brownie", "Vegan Brownie",
    "Vanilla Cupcake", "Gluten Free Vanilla Cupcake", "Vegan Vanilla Cupcake", 
    "Chocolate Cupcake", "Vegan Chocolate Cupcake", "Gluten Free Chocolate Cupcake", 
    "Chocolate Chip Cookies", "Vegan Chocolate Chip Cookie", "Gluten Free Chocolate Chip Cookie"
]

   
input_type = st.selectbox("Select a baked good:", baked_goods_options)

gluten_free_options = [ 
    "Gluten Free Brownie", "Gluten Free Vanilla Cupcake","Gluten Free Chocolate Cupcake","Gluten Free Chocolate Chip Cookie"]

chef_tips = {
   "Brownie":["nuts", "Oreos", "espresso powder", "caramel", "Cheesecake"],
    "Gluten Free Brownie":["Dried Fruit", "White chocolate chips", "Jam", "candied fruit"],  
    "Vegan Brownie":["Peanut butter", "Coconut flakes", "crushed pretzels"] ,
    "Vanilla Cupcake":["Sprinkles", "Lemon zest", " chocolate chips", "ganache"] ,
    "Vegan Vanilla Cupcake": ["Sprinkles", "Lemon zest", " chocolate chips", "coconut", "", "pistachio butter"],
    "Gluten Free Vanilla Cupcake":["Fruit compote"," caramel ganache","passion fruit curd", "chcolate drizzle"] ,
    "Chocolate Cupcake":["orange zest", "nutella", "baileys","crushed pepermint" ] ,
    "Vegan Chocolate Cupcake": ["vegan sprinkles", "jams", "vegan marshmallows","praline"],
    "Gluten Free Chocolate Cupcake":["strawberry compote", "vanilla paste", "frangipane"," chocolate covered strawberries"] ,
    "Chocolate Chip Cookies":["flaky salt", " toffee", "shaved chocolate", "crushed potato chips", "butterscotch", "chocolate candies"] ,
    "Vegan Chocolate Chip Cookie":["flaky salt", "almond slivers", "dried cranberries", "candied ginger"] ,
    "Gluten Free Chocolate Chip Cookie":["flaky salt", "smoked salt", "pumpkin spice", "raisins", "chocolate covered fruit"] 
}

if input_type in baked_goods_images:
    st.image(baked_goods_images[input_type], caption=input_type, use_container_width=True)
else:
    st.warning("Image not available for this baked good.")

filtered_dataset = baked_goods_dataset.loc[baked_goods_dataset['type'] == input_type].copy()
filtered_dataset['flour_ratio'] = filtered_dataset['flour_ratio'].astype(float)
scaler = StandardScaler()
normalized_ratios = scaler.fit_transform(filtered_dataset[['flour_ratio', 'sugar_ratio', 'Butter_ratio']])
print(filtered_dataset[['flour_ratio', 'sugar_ratio', 'Butter_ratio']].isnull().sum())
print(filtered_dataset[['flour_ratio', 'sugar_ratio', 'Butter_ratio']].dtypes)
print(filtered_dataset['flour_ratio'].unique())


filtered_dataset[['flour_ratio', 'sugar_ratio', 'Butter_ratio']] = filtered_dataset[
    ['flour_ratio', 'sugar_ratio', 'Butter_ratio']
].astype(float)


if filtered_dataset.empty:
    st.error("No data available for the specified baked good type.")
else:
    filtered_dataset['texture_category_list'] = filtered_dataset['texture_category'].apply(lambda x: x.split(','))
    unique_textures = sorted(set([texture for texture_list in filtered_dataset['texture_category_list'] for texture in texture_list]))
    input_texture = st.selectbox("Select a desired texture:", unique_textures)

    mlb = MultiLabelBinarizer()
    texture_encoded = mlb.fit_transform(filtered_dataset['texture_category_list'])
    texture_encoded_df = pd.DataFrame(texture_encoded, columns=mlb.classes_)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    type_encoded = encoder.fit_transform(filtered_dataset[['type']])

    scaler = StandardScaler()
    normalized_ratios = scaler.fit_transform(filtered_dataset[['flour_ratio', 'sugar_ratio', 'Butter_ratio']])

    X = np.hstack((type_encoded, texture_encoded, normalized_ratios))
    y = filtered_dataset[['flour_ratio', 'sugar_ratio', 'Butter_ratio']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    random_forest_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    random_forest_model.fit(X_train, y_train)

    y_pred_rf = random_forest_model.predict(X_test)
    r2_flour_rf = r2_score(y_test[:, 0], y_pred_rf[:, 0])
    r2_sugar_rf = r2_score(y_test[:, 1], y_pred_rf[:, 1])
    r2_butter_rf = r2_score(y_test[:, 2], y_pred_rf[:, 2])

    st.write(f"Flour Ratio R^2: {r2_flour_rf}")
    st.write(f"Sugar Ratio R^2: {r2_sugar_rf}")
    st.write(f"Butter Ratio R^2: {r2_butter_rf}")

    flour_cups = st.number_input("Enter the flour amount in cups:", min_value=0.0, step=0.1)
    sugar_cups = st.number_input("Enter the sugar amount in cups:", min_value=0.0, step=0.1)
    butter_cups = st.number_input("Enter the butter amount in cups:", min_value=0.0, step=0.1)

    unit = st.selectbox("Select unit of measurement:", ["Cups", "Grams"])


    total_cups = flour_cups + sugar_cups + butter_cups
    if total_cups > 0:
        flour_ratio = flour_cups / total_cups
        sugar_ratio = sugar_cups / total_cups
        butter_ratio = butter_cups / total_cups

        example_base_ratios = scaler.transform([[flour_ratio, sugar_ratio, butter_ratio]])
        user_input_texture = mlb.transform([[input_texture]])
        user_input = np.hstack((type_encoded[0], user_input_texture[0], example_base_ratios[0]))
        user_input = user_input.reshape(1, -1)

        predicted_ratios = random_forest_model.predict(user_input)
        optimized_ratios = scaler.inverse_transform(predicted_ratios)[0]

        optimized_flour_cups = optimized_ratios[0] * total_cups / sum(optimized_ratios)
        optimized_sugar_cups = optimized_ratios[1] * total_cups / sum(optimized_ratios)
        optimized_butter_cups = optimized_ratios[2] * total_cups / sum(optimized_ratios)

        if unit == "Cups":
            st.write(f"Optimized Flour Measurement: {optimized_flour_cups:.2f} cups")
            st.write(f"Optimized Sugar Measurement: {optimized_sugar_cups:.2f} cups")
            st.write(f"Optimized Butter Measurement: {optimized_butter_cups:.2f} cups")
        else:
            conversion_factors = {"Flour": 120, "Sugar": 200, "Butter": 227}
            optimized_flour_grams = optimized_flour_cups * conversion_factors["Flour"]
            optimized_sugar_grams = optimized_sugar_cups * conversion_factors["Sugar"]
            optimized_butter_grams = optimized_butter_cups * conversion_factors["Butter"]
            st.write(f"Optimized Flour Measurement: {optimized_flour_grams:.2f} grams")
            st.write(f"Optimized Sugar Measurement: {optimized_sugar_grams:.2f} grams")
            st.write(f"Optimized Butter Measurement: {optimized_butter_grams:.2f} grams")
        
        if input_type in chef_tips: 
            st.subheader( "Chef tips for added indulgences")
            mix_in_tips = ",".join(chef_tips[input_type])
            st.write(f"Enhance your {input_type.lower()}")


        
        fig, ax = plt.subplots(3, 1, figsize=(12, 18))
        sns.histplot(y_test[:, 0], color='blue', label='Actual Flour Ratio', kde=True, stat="density", ax=ax[0])
        sns.histplot(y_pred_rf[:, 0], color='red', label='Predicted Flour Ratio', kde=True, stat="density", ax=ax[0])
        ax[0].set_xlabel('Flour Ratio')
        ax[0].set_ylabel('Density')
        ax[0].set_title('Distribution of Actual vs. Predicted Flour Ratios')
        ax[0].legend()

        sns.histplot(y_test[:, 1], color='blue', label='Actual Sugar Ratio', kde=True, stat="density", ax=ax[1])
        sns.histplot(y_pred_rf[:, 1], color='red', label='Predicted Sugar Ratio', kde=True, stat="density", ax=ax[1])
        ax[1].set_xlabel('Sugar Ratio')
        ax[1].set_ylabel('Density')
        ax[1].set_title('Distribution of Actual vs. Predicted Sugar Ratios')
        ax[1].legend()

        sns.histplot(y_test[:, 2], color='blue', label='Actual Butter Ratio', kde=True, stat="density", ax=ax[2])
        sns.histplot(y_pred_rf[:, 2], color='red', label='Predicted Butter Ratio', kde=True, stat="density", ax=ax[2])
        ax[2].set_xlabel('Butter Ratio')
        ax[2].set_ylabel('Density')
        ax[2].set_title('Distribution of Actual vs. Predicted Butter Ratios')
        ax[2].legend()

        st.pyplot(fig)

        texture_counts = baked_goods_dataset.groupby(['type', 'texture_category']).size().reset_index(name='counts')



