import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import math
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.model_selection import cross_val_score

data = pd.read_csv("dataset.csv")
df = data.copy()

# הכנת הנתונים
prepared_df = prepare_data(df)

# הגדרת מילון איזורי החיוג
area_to_dial_code = {
    '03': [
        'פתח תקוה והסביבה', 'רמת', 'רמת גן - גבעתיים', 'תל אביב', 'בקעת אונו', 
        'מושבים במרכז', 'תל', 'פתח'
    ],
    '02': [
        'ירושלים והסביבה', 'בית שמש והסביבה', 'ירושלים', 'יישובי השומרון'
    ],
    '04': [
        'חיפה וחוף הכרמל', 'מושבים', 'כרמיאל והסביבה', 'גליל ועמקים', 
        'עכו - נהריה', 'טבריה והסביבה', 'קריות', 'עמק יזרעאל', 'גליל', 
        'פרדס', 'חדרה וישובי עמק חפר', 'קיסריה והסביבה', 'טבריה', 
        'זכרון - בנימינה', 'עמק', 'פרדס חנה - כרכור', 'מושבים בצפון', 'חיפה'
    ],
    '08': [
        'נס ציונה - רחובות', 'ראשל"צ והסביבה', 'חולון - בת ים', 'באר שבע והסביבה', 
        'גדרה יבנה והסביבה', 'אשדוד - אשקלון', 'מודיעין והסביבה', 'חולון', 
        'מושבים בשפלה', 'אילת והערבה', 'רמלה - לוד', 'מודיעין', 'רמלה', 
        'ראשל"צ', 'נס', 'מושבים בדרום', 'רחובות'
    ],
    '09': [
        'רעננה - כפר סבא', 'מושבים בשרון', 'ראש העין והסביבה', 'נתניה והסביבה', 
        'הוד השרון והסביבה', 'רמת השרון - הרצליה', 'אזור השרון והסביבה', 
        'רעננה', 'הוד', 'נתניה', 'ראש'
    ]
}

# פונקציה למיפוי איזורי חיוג
def map_area_to_dial_code(area):
    for dial_code, areas in area_to_dial_code.items():
        if area in areas:
            return dial_code
    return '05'

def prepare_data(data):
    # סידור - הורדת העמודות עם יותר מחצי ערכים חסרים
    data = data.drop(columns=['Test', 'Supply_score'], errors='ignore')
    
    # רשימת הערכים להחלפה
    values_to_replace = ['None', 'nan', 'לא מוגדר']

    # מעבר על כל התאים בכל העמודות והחלפת הערכים
    for col in data.columns:
        data[col] = data[col].apply(lambda x: None if x in values_to_replace else x)
    data = data.applymap(lambda x: np.nan if pd.isna(x) else x)
    
    ######################################################################################################################
    # סידור עמודות פר עמודה 
    
    # manufactor - במידה ויהיו ערכים חסרים נשלים אותם על ידי זיהוי המודל.
    data.loc[:, 'manufactor'] = data['manufactor'].replace('Lexsus', 'לקסוס')
    data['manufactor'] = data.groupby('model')['manufactor'].transform(lambda x: x.fillna(x.mode()[0]))
    
    # Gear - בקירוב ל-2005 התחילו לייצר מכוניות אוטומטיות במקום עם הילוכים. לכן אם יהיה לנו ערך חסר זה ילך על פי העובדה הזאת.
    data.loc[:, 'Gear'] = data['Gear'].replace('אוטומט', 'אוטומטית')
    data['Gear'] = data.apply(lambda row: 'אוטומטית' if pd.isna(row['Gear']) and row['Year'] >= 2005 else 
                    'ידנית' if pd.isna(row['Gear']) and row['Year'] < 2005 else row['Gear'], axis=1)
    
    # capacity_Engine - הפיכת ערכים למספר + אם זה קטן מ-400 ערך הכי קטן למנוע אז להכפיל ב100 + מילוי על ידי חציון
    for idx in data.index:
        value = data.loc[idx, 'capacity_Engine']
        if isinstance(value, str):
            clean_value = value.replace(',', '')
            data.loc[idx, 'capacity_Engine'] = int(clean_value)
    
    # הכפלת ערכים קטנים מ-400 ב-100
    for idx in data.index:
        value = data.loc[idx, 'capacity_Engine']
        if value < 400:
            data.loc[idx, 'capacity_Engine'] = value * 100

    # חישוב החציוני של העמודה ומילוי הערכים החסרים
    median_value = data['capacity_Engine'].median()
    data['capacity_Engine'].fillna(median_value, inplace=True)
    
    # ניקוי עמודת 'City'
    # מילון המרה לערים באנגלית לעברית
    translations = {
        'Tel aviv': 'תל אביב',
        'Rehovot': 'רחובות',
        'haifa': 'חיפה',
        'jeruslem': 'ירושלים',
        'ashdod': 'אשדוד',
        'Tzur Natan': 'צור נתן'
    }
    
    # פונקציה להמרת ערים באנגלית לעברית
    def translate_city(city):
        return translations.get(city, city)
    
    # המרת ערים באנגלית לעברית
    data['City'] = data['City'].apply(translate_city)
    
    # איחוד "תל אביב יפו" ל"תל אביב"
    data['City'] = data['City'].replace('תל אביב יפו', 'תל אביב')
    
    # החלפת "הוד" ל"הוד השרון"
    data['City'] = data['City'].replace('הוד', 'הוד השרון')
    
    # מילוי ערכים חסרים בעמודת 'City' לפי ערך שכיח
    data['City'] = data.groupby('City')['City'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
    
    # עיבוד עמודת 'Area'
    data['Area'] = data.groupby('City')['Area'].transform(lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else x)
    data['City'] = data['City'].astype(str)
    data['Area'] = data['Area'].astype(str)
    
    # מיפוי עמודת 'Area' לאיזורי חיוג
    data['Dial_code'] = data['Area'].apply(map_area_to_dial_code)
    
    # סידור עמודת 'Engine_type'
    data.loc[:, 'Engine_type'] = data['Engine_type'].replace('היבריד', 'היברידי')
    
    # מילוי ערכים חסרים בעמודת Engine_type לפי ערך שכיח
    if 'Engine_type' in data.columns:
        most_common_engine_type = data['Engine_type'].mode()[0] if not data['Engine_type'].mode().empty else np.nan
        data['Engine_type'].fillna(most_common_engine_type, inplace=True)
    
    # מניפולציות על עמודת Km
    for idx in data.index:
        value = data.loc[idx, 'Km']
        if isinstance(value, str):
            # הסרת פסיקים והמרה למספר שלם
            clean_value = value.replace(',', '')
            data.loc[idx, 'Km'] = int(clean_value)
        
        # הכפלת ערכים קטנים מ-1000 ב-1000
        if isinstance(data.loc[idx, 'Km'], int) and data.loc[idx, 'Km'] < 1000:
            data.loc[idx, 'Km'] *= 1000
    
    # טיפול בערכים חסרים בעמודת Km על פי הנוסחה
    current_year = 2024
    data['Km'] = data.apply(
        lambda row: (current_year - row['Year']) * 12000 if pd.isna(row['Km']) and pd.notna(row['Year']) else row['Km'],
        axis=1)
    
    # Color - הסרת מילים שלא תורמות + השלמת ערכים על ידי ערך שכיח
    if 'Color' in data.columns:
        words_to_remove = ["פנינה", "בהיר", "כהה", "מטאלי", "מטלי"]
        for idx in data.index:
            value = data.loc[idx, 'Color']
            if isinstance(value, str):
                words = value.split()
                filtered_words = [word for word in words if word not in words_to_remove]
                data.loc[idx, 'Color'] = ' '.join(filtered_words)
    
    # מציאת הצבע השכיח ביותר
    most_common_color = data['Color'].mode()[0] if not data['Color'].mode().empty else np.nan
    
    # מילוי ערכים חסרים בצבע השכיח ביותר
    data['Color'].fillna(most_common_color, inplace=True)
    
    # מילוי ערכים חסרים בעמודות Prev_ownership ו-Curr_ownership
    if 'Prev_ownership' in data.columns:
        data['Prev_ownership'].fillna('אחר', inplace=True)
    
    if 'Curr_ownership' in data.columns:
        data['Curr_ownership'].fillna('אחר', inplace=True)
    
    # הסרת העמודות המיותרות
    columns_to_drop = ['model', 'Area', 'Pic_num', 'Cre_date', 'Repub_date', 'Description']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
    
    return data
