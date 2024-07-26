import pandas as pd
df = pd.read_csv('rodent_edited5.csv')

#take out last 3 lines for no dates

passed_df = df[df['RESULT'] == 'Passed']
coordinates_passed = passed_df['LOCATION'].str.extract(r'\(([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)')
coordinates_passed.columns = ['LATITUDE', 'LONGITUDE']
passed_df['DATE'] = pd.to_datetime(passed_df['APPROVED_DATE'])
coordinates_passed['DATE'] = passed_df['DATE'].sort_values(ascending=True).reset_index(drop=True)
print(coordinates_passed)

rat_activity_df = df[df['RESULT'] == 'Rat Activity']
coordinates_rat_activity = rat_activity_df['LOCATION'].str.extract(r'\(([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)')
coordinates_rat_activity.columns = ['LATITUDE', 'LONGITUDE']
rat_activity_df['DATE'] = pd.to_datetime(rat_activity_df['APPROVED_DATE'])
coordinates_rat_activity['DATE'] = rat_activity_df['DATE'].sort_values(ascending=True).reset_index(drop=True)
print(coordinates_rat_activity)

stoppagedone_df = df[df['RESULT'] == 'Stoppage done']
coordinates_stoppagedone = stoppagedone_df['LOCATION'].str.extract(r'\(([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)')
coordinates_stoppagedone.columns = ['LATITUDE', 'LONGITUDE']
stoppagedone_df['DATE'] = pd.to_datetime(stoppagedone_df['APPROVED_DATE'])
coordinates_stoppagedone['DATE'] = stoppagedone_df['DATE'].sort_values(ascending=True).reset_index(drop=True)
print(coordinates_stoppagedone)

monitoringVisit_df = df[df['RESULT'] == 'Monitoring visit']
coordinates_monitoringvisit = monitoringVisit_df['LOCATION'].str.extract(r'\(([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)')
coordinates_monitoringvisit.columns = ['LATITUDE', 'LONGITUDE']
monitoringVisit_df['DATE'] = pd.to_datetime(monitoringVisit_df['APPROVED_DATE'])
coordinates_monitoringvisit['DATE'] = monitoringVisit_df['DATE'].sort_values(ascending=True).reset_index(drop=True)
print(coordinates_monitoringvisit)

failedOther_df = df[df['RESULT'] == 'Failed for Other R']
coordinates_failedOther = failedOther_df['LOCATION'].str.extract(r'\(([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)')
coordinates_failedOther.columns = ['LATITUDE', 'LONGITUDE']
failedOther_df['DATE'] = pd.to_datetime(failedOther_df['APPROVED_DATE'])
coordinates_failedOther['DATE'] = failedOther_df['DATE'].sort_values(ascending=True).reset_index(drop=True)
print(coordinates_failedOther)

baitApplied_df = df[df['RESULT'] == 'Bait applied']
coordinates_baitApplied = baitApplied_df['LOCATION'].str.extract(r'\(([-+]?\d+\.\d+), ([-+]?\d+\.\d+)\)')
coordinates_baitApplied.columns = ['LATITUDE', 'LONGITUDE']
baitApplied_df['DATE'] = pd.to_datetime(baitApplied_df['APPROVED_DATE'])
coordinates_baitApplied['DATE'] = baitApplied_df['DATE'].sort_values(ascending=True).reset_index(drop=True)
print(coordinates_baitApplied)