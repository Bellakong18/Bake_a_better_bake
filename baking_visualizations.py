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


baked_goods_dataset = pd.read_csv('categorized_baked_goods_dataset_final.csv')

baked_good_types = baked_goods_dataset['type'].unique()
texture_types = {
    'gooey' :[ 'gooey', 'fudgey', 'molten', 'chewy' 'thick fudgy', 'moist', 'rich','fudgy center', 'super fudgy', 'fudgy', 'chewy nutty','chewy chocolate chunks'], 
    'cakey':[ 'cakey', 'crumbly', 'dense', 'set cracks'], 
    'soft': ['soft', 'tender', 'fluffy'], 
    'firm': ['firm', 'crisp', 'crunchy', 'hard','slight jiggle'], 
    'chewy': ['chewy', 'golden brown'], 
    'crispy': ['crispy', 'golden and crispy', ], 
    'thin': ['thin','Crinkled edges'], 
    'thick':['thick', 'dense','thicker cookies'],
    'light and fluffy': ['light', 'fluffy', 'light and fluffy'], 
    'moist': ['tender', 'moist', ],
    'creamy': ['creamy', ''], 
    'rich': ['rich','dense', 'decadent']

}
print(baked_goods_dataset[baked_goods_dataset['texture_category'].str.contains('light and fluffy', na=False)])
for main_category, keywords in texture_types.items():
    print(main_category, keywords)
    print(texture_types['light and fluffy'])

def map_to_main_category(texture):
    if isinstance(texture, str):
        for main_category, keywords in texture_types.items():
            if any(keyword == texture.strip() for keyword in keywords):
                return main_category
            if any(keyword in texture for keyword in keywords):
                return main_category
    return 'uncategorized'

baked_goods_dataset['main_texture_category'] = baked_goods_dataset['texture_category'].apply(map_to_main_category)
selected_baked_goods = ['Brownie', 'Chocolate Chip Cookie', 'Cupcake']
filtered_baked_goods = baked_goods_dataset[
    baked_goods_dataset['type'].str.contains('|'.join(selected_baked_goods), case=False)
]
custom_palette = sns.color_palette(["#FF69B4", "#BA55D3", "#9400D3", "#8A2BE2", "#DDA0DD"])

for baked_good in selected_baked_goods:
    subset = filtered_baked_goods[filtered_baked_goods['type'].str.contains(baked_good, case=False)]
    
    main_texture_counts = subset['main_texture_category'].value_counts().reset_index()
    main_texture_counts.columns = ['main_texture_category', 'counts']
    
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(data=main_texture_counts, x='main_texture_category', y='counts', palette=sns.color_palette("Purples", len(main_texture_counts)))
    plt.title(f'Main Texture Distribution for {baked_good} and Variations', fontsize=16)
    plt.xlabel('Main Texture Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    
    for index, row in main_texture_counts.iterrows():
        barplot.text(index, row['counts'] + 0.1, int(row['counts']), color='black', ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    for index, row in main_texture_counts.iterrows():
        barplot.text(index, row['counts'] + 0.1, int(row['counts']), color='black', ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.show()