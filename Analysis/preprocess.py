import pandas as pd
import numpy as np

raw = pd.read_csv('../Data/EdmontonRealEstateData_train.csv')

Edmonton = raw.drop([
    'house_suit',
    'house_number',
    'house_suff',
    'street_name',
    'city',
    'full_address',
    'fully_taxable',
    'display_type',
    'geometry',
    'result_code',
    'build_year_mbc',
    'landuse_description',
    "postal_code",
    "site_coverage",
    "market_building_class",
    "fully_complete",
    "effective_build_year",
    "neighbourhood"
], axis=1)

# just keep m^2 value for tot_gross_area_description
Edmonton['tot_gross_area_description'] = Edmonton['tot_gross_area_description'].astype(str)
Edmonton['tot_gross_area_description'] = Edmonton['tot_gross_area_description'].str.split(' ',1,expand=True)[0]
Edmonton['tot_gross_area_description'] = Edmonton['tot_gross_area_description'].astype('float')

# just keep m^2 value lot_size
Edmonton['lot_size'] = Edmonton['lot_size'].astype(str)
Edmonton['lot_size'] = Edmonton['lot_size'].str.split(' ',1,expand=True)[0]
Edmonton['lot_size'] = Edmonton['lot_size'].astype('float')

# encode some categorical variables
Edmonton['basement_finished'] = Edmonton['basement_finished'].map({'Yes': 1, 'NO': 0})
Edmonton['has_garage'] = Edmonton['has_garage'].map({'Yes': 1, 'NO': 0})
Edmonton['has_fireplace'] = Edmonton['has_fireplace'].map({'Yes': 1, 'NO': 0})
Edmonton['walkout_basement'] = Edmonton['walkout_basement'].map({'Yes': 1, 'NO': 0})
Edmonton['air_conditioning'] = Edmonton['air_conditioning'].map({'Yes': 1, 'NO': 0})

# categorize valuation group
Edmonton['valuation_group'] = Edmonton['valuation_group'].map({
    'RESIDENTIAL SOUTH': 0,
    'RESIDENTIAL NORTH': 1,
    'RESIDENTIAL WC': 2,
    'RESIDENTIAL RIVVAL': 3,
    'RESIDENTIAL LAND': 4,
    'SPECIAL PURPOSE': 5,
    'LAND': 6
})

# get rid of unnamed column
Edmonton = Edmonton.iloc[:,1:]

#Edmonton['assessed_value'] = np.log(Edmonton['assessed_value'])
#Edmonton['lot_size'] = np.log(Edmonton['lot_size'])


Edmonton.to_csv("preprocessed_train.csv")

