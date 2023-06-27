from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
House_df = pd.read_csv('Clean_data_ (1).csv')
cors = CORS(app)

model = pickle.load(open('XGRB.pkl', 'rb'))


@app.route('/', methods=['GET', 'POSt'])
def index():
    Cetnral_Air = sorted(House_df['CentralAir'].unique())
    Heating = sorted(House_df['Heating'].unique())
    Neighborhood = sorted(House_df['Neighborhood'].unique())
    GarageType = sorted(House_df['GarageType'].unique())
    GarageCar = sorted(House_df['GarageCars'].unique())
    KitchenAbvGr = sorted(House_df['KitchenAbvGr'].unique())
    YearBuilt = sorted(House_df['YearBuilt'].unique(), reverse=True)
    FullBath = sorted(House_df['FullBath'].unique())
    #
    print(Cetnral_Air)
    return render_template('index.html', Cetnral_Air=Cetnral_Air, Heating=Heating, Neighborhood=Neighborhood,
                           GarageType=GarageType, GarageCar=GarageCar, KitchenAbvGr=KitchenAbvGr, YearBuilt=YearBuilt,
                           FullBath=FullBath)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    user_input = ['CentralAir', 'Heating', 'Neighborhood', 'GarageType', 'GarageCars', 'KitchenAbvGr', 'YearBuilt', 'FullBath']
    CentralAir = request.form.get('CentralAir')
    Heating = request.form.get('Heating')
    Neighborhood = request.form.get('Neighborhood')
    GarageType = request.form.get('GarageType')
    GarageCar = request.form.get('GarageCar')
    KitchenAbvGr = request.form.get('KitchenAbvGr')
    YearBuilt = request.form.get('YearBuilt')
    FullBath = request.form.get('FullBath')

    random_cols = ['Id', 'MSSubClass', 'MSZoning', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities',
                   'LotConfig',
                   'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallCond', 'YearRemodAdd',
                   'RoofStyle',
                   'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond',
                   'Foundation',
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                   'BsmtUnfSF',
                   'TotalBsmtSF', 'HeatingQC', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                   'BsmtFullBath',
                   'BsmtHalfBath', 'BedroomAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces',
                   'GarageYrBlt',
                   'GarageFinish', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
                   'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
                   'SaleCondition']
    data = {}
    for col in House_df.columns:
        if col in user_input:
            data[col] = [request.form.get(col)]
        elif House_df[col].dtype.kind in 'inf':
            data[col] = [np.random.uniform(House_df[col].min(), House_df[col].max())]
        else:
            values = House_df[col].dropna().unique()
            probabilities = House_df[col].dropna().value_counts(normalize=True)
            data[col] = [np.random.choice(values, p=probabilities)]
    df = pd.DataFrame(data)

    for col in random_cols:
        if col not in df.columns:
            df[col] = np.random.uniform(House_df[col].min(), House_df[col].max())

    price = model.predict(df)
    response = {
        'prediction': price
    }

    return json.dumps(response)


if __name__ == '__main__':
    app.run(debug=True)
