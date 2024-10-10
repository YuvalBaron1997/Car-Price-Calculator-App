from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import math
import pandas as pd
import pickle

# קריאת הנתונים
data = pd.read_csv("dataset.csv")
df = data.copy()

# הכנת הנתונים
prepared_df = prepare_data(df)

# פיצול הנתונים לסט אימון וסט בדיקה
X = prepared_df.drop(columns=['Price'])
y = prepared_df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# הגדרת הטרנספורמציה לעמודות הקטגוריאליות עם טיפול בערכים לא ידועים
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), 
         X.select_dtypes(include=['object', 'category']).columns)
    ],
    remainder='passthrough'  # השארת העמודות המספריות ללא שינוי
)

# יצירת ה-Pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('E', ElasticNet())
])

# הגדרת Grid של פרמטרים לחיפוש
param_grid = {
    'E__alpha': [0.05, 0.06, 0.07, 0.09, 0.1, 0.5, 1.0, 5.0, 5.5],
    'E__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# יצירת אובייקט GridSearchCV עם קיפולים
grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='neg_mean_squared_error', cv=10)

# התאמת GridSearchCV על נתוני האימון
grid_search.fit(X_train, y_train)

# שליפת הפרמטרים והמודל האופטימליים
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# חיזוי על נתוני הבדיקה
y_pred = best_model.predict(X_test)

# שמירה לקובץ
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# חישוב והדפסת ה-RMSE
#mse = mean_squared_error(y_test, y_pred)
#rmse = math.sqrt(mse)
#print(f'RMSE: {rmse}')

# הדפסת הפרמטרים האופטימליים
#print(f'Best alpha: {best_params["E__alpha"]}')
#print(f'Best l1_ratio: {best_params["E__l1_ratio"]}')