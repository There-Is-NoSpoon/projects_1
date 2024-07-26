import numpy as np
import pandas as pd
from pyproj import Transformer

def get_shooting_data(fname):
    df = pd.read_csv(fname)

    # proj = Transformer.from_crs(2263, 4326)
    # lat = df.apply(lambda row: proj.transform(row.X_COORD_CD, row.Y_COORD_CD)[0], axis=1)
    # long = df.apply(lambda row: proj.transform(row.X_COORD_CD, row.Y_COORD_CD)[1], axis=1)
    lat = df.X_COORD_CD
    long = df.Y_COORD_CD

    datetime = pd.to_datetime(df.OCCUR_DATE) + pd.to_timedelta(df.OCCUR_TIME)

    out = {
        "latitude": lat,
        "longitude": long,
        "datetime": datetime
    }

    return pd.DataFrame(out)

def get_rat_data(fname):
    df = pd.read_csv(fname)
    coordinates_passed = df['LOCATION'].str.extract(r'\(([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)')
    coordinates_passed.columns = ['latitude', 'longitude']
    coordinates_passed = coordinates_passed.astype("float")
    coordinates_passed['datetime'] = pd.to_datetime(df['APPROVED_DATE'])

    coordinates_passed.latitude = df.X_COORD
    coordinates_passed.longitude = df.Y_COORD
    coordinates_passed.latitude.replace(0, np.nan, inplace=True)
    coordinates_passed.longitude.replace(0, np.nan, inplace=True)

    return coordinates_passed

    