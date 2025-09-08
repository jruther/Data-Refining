import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from windrose import WindroseAxes
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
%matplotlib inline


max_power_value = 2350


data = 'data'
img = 'img'

Info = pd.read_csv(data + "\Information_7632_2023-01-01_2023-12-31.csv", delimiter = ";",parse_dates=[['Date', 'Time']],decimal =',')

ST = pd.read_csv(data + "\Status_7632_2023-01-01_2023-12-31.csv", delimiter = ";", decimal =',', thousands='.')

wf = pd.read_csv(data +"\WEC_10 minutes_7632_2023-01-01_2023-12-31.csv", delimiter = ";", decimal =',', thousands='.')

temp1 = pd.read_csv(data + "\Temp. 1 CS82a_10 minutes_7632_2023-01-01_2023-12-31.csv", delimiter = ";", decimal =',', thousands='.')

wfd = pd.read_csv(data + "\WEC_Day_7632_2023-01-01_2023-12-31.csv", delimiter = ";", decimal =',', thousands='.')

pd.options.display.max_columns = None

temp1['Datetime']=pd.to_datetime(temp1['Time'])
wf['Datetime']=pd.to_datetime(wf['Time'])

ST['Datetime'] =pd.to_datetime(ST['Date'] + ' '+ ST['Time'])

# Merge 'temp1' columns into 'wf' based on 'Datetime' and 'Plant' columns
wf = wf.merge(temp1, left_on=['Datetime', 'Plant'], right_on=['Datetime', 'Plant'], how='left')

wf.columns = [col.strip() for col in wf.columns]

#wf

#wf = wf.loc[wf[' Power Ø [kW]']>=0]

wf.columns

wf1 = wf[['Datetime','Plant', 'Wind Ø [m/s]', 'Rotation speed Ø [1/min]',
       'Power Ø [kW]','Power Avail. wind. Ø [kW]', 'Power Avail. techn. Ø [kW]',
       'Power Avail. force maj. Ø [kW]', 'Power Avail. ext. Ø [kW]', 'Energy prod. [kWh].1',
       'Blade angle Ø [°]', 'Nacelle position [°]',
       'Temp. Nacelle outside 1 [°C]']]

### Now we will filter all the data

# Note for self.  Match all data with status codes and display!!
#  https://renewable-analytics.netlify.app/2018/02/12/wind-turbine-power-curve-status-and-statistical-filtering/

ST.head()

ST.tail()

ST['DateTime'] = ST['Date'] + ' '+ ST['Time']

ST.drop(['Date', 'Time'], axis=1)

ST['Datetime']=pd.to_datetime(ST['DateTime'])

ST = ST.set_index('Datetime')
ST = ST.drop(['DateTime', 'Date', 'Time'], axis=1)
ST

ST_Main = ST[['Plant', 'Main status', 'Additional status']]

ST_Main

ST_Main = ST_Main[~ST_Main.index.duplicated(keep= 'first')]

ST_Main

ST_Main = ST_Main.resample('1min').ffill()

ST_Main.to_csv('Main_STatus_output.csv')

####  OK, I'm taking the csv from above to analyse it separately.

Op = ST_Main.loc[(ST_Main['Main status'] == 0.0) & (ST_Main['Additional status'] == 0.0)]

Op

Op1 = Op.loc[Op['Plant']==1]
Op2 = Op.loc[Op['Plant']==2]
Op3 = Op.loc[Op['Plant']==3]
Op4 = Op.loc[Op['Plant']==4]

Op1

Op1['Counter']=1
Op2['Counter']=1
Op3['Counter']=1
Op4['Counter']=1

#Op2.to_csv('Op2_status.csv')


# Now we will extract the External Set Point information dataset.
Info.head()

SP = Info.loc[Info['Plant']==6]

SP.head()

SP.tail()

SP.columns

SP = SP[['Date_Time', 'Plant','Main information','Additional information','Information Text']]
SP

SP = SP.loc[SP['Main information'] == 221]

SP

SP = SP.loc[SP['Additional information'] != 110]

SP

SP = SP[['Date_Time', 'Additional information']]
SP

SP.columns = ['Date_Time','Set_Point']



SP = SP.set_index(pd.DatetimeIndex(SP['Date_Time']))

SP=SP.drop(['Date_Time'], axis=1)

SP = SP.resample('1min').ffill()
SP

SP100 = SP.loc[SP['Set_Point'] == 100.0]

SP100

SP100.describe()

SP100['Counter']=1

SP100

SP100s = SP100.groupby(pd.Grouper(freq='10min')).sum()
SP100s



SP100m=SP100.groupby(pd.Grouper(freq='10min')).mean()
SP100m

SP100 = pd.merge(SP100m,SP100s, left_index=True, right_index=True)
SP100

SP100 = SP100[['Set_Point_x', 'Counter_y']]

SP100

SP100 = SP100.loc[SP100['Counter_y'] ==10]
SP100

SP100 = SP100.shift(periods=1, freq='10min')

SP100

SP100.columns = ['Set_Point', 'Count']
SP100

#Op1.to_csv('check-number-before-group.csv')

Op1=Op1.groupby(pd.Grouper(freq='10min')).sum()
Op2=Op2.groupby(pd.Grouper(freq='10min')).sum()
Op3=Op3.groupby(pd.Grouper(freq='10min')).sum()
Op4=Op4.groupby(pd.Grouper(freq='10min')).sum()

Op1

Op1 = Op1.shift(periods=1, freq='10min')
Op2 = Op2.shift(periods=1, freq='10min')
Op3 = Op3.shift(periods=1, freq='10min')
Op4 = Op4.shift(periods=1, freq='10min')

Op1

Op1 = Op1.loc[Op1['Counter']==10]
Op2 = Op2.loc[Op2['Counter']==10]
Op3 = Op3.loc[Op3['Counter']==10]
Op4 = Op4.loc[Op4['Counter']==10]

df1 = pd.merge(SP100,Op1, left_index=True, right_index=True)
df2 = pd.merge(SP100,Op3, left_index=True, right_index=True)
df3 = pd.merge(SP100,Op3, left_index=True, right_index=True)
df4 = pd.merge(SP100,Op4, left_index=True, right_index=True)
#df.to_csv('all_data.csv')

df1

T1.columns

T1

T1_Full = pd.merge(T1,df1, left_index=True, right_index=True)
T2_Full = pd.merge(T2,df2, left_index=True, right_index=True)
T3_Full = pd.merge(T3,df3, left_index=True, right_index=True)
T4_Full = pd.merge(T4,df4, left_index=True, right_index=True)


T1_Full.describe()

T1_Full = T1_Full.round(decimals=1)
T2_Full = T2_Full.round(decimals=1)
T3_Full = T3_Full.round(decimals=1)
T4_Full = T4_Full.round(decimals=1)

T1_Full.describe()



plt.plot(T1_Full['Wind Ø [m/s]'],T1_Full['Power Ø [kW]'], 'bo')
plt.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('kW', color ='r', fontsize = 20)
plt.xlabel('Windspeed m/s', color = 'r', fontsize = 20)
plt.title('T1_Full 10-minute Averaged Power Values', color = 'r', fontsize = 20)
plt.legend(['Power Ø','Power Curve'], prop={'size': 15})
plt.grid()
plt.rcParams["figure.figsize"] = [20,10]
#plt.savefig('img/Turbine_1_Power_Curve.jpg')
plt.show()





plt.plot(T2_Full['Wind Ø [m/s]'],T2_Full['Power Ø [kW]'], 'bo')
plt.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('kW', color ='r', fontsize = 20)
plt.xlabel('Windspeed m/s', color = 'r', fontsize = 20)
plt.title('T2_Full 10-minute Averaged Power Values', color = 'r', fontsize = 20)
plt.legend(['Power Ø','Power Curve'], prop={'size': 15})
plt.grid()
plt.rcParams["figure.figsize"] = [20,10]
#plt.savefig(MONTH + '/Images/Turbine_1_Power_Curve.jpg')
plt.show()

plt.plot(T3_Full['Wind Ø [m/s]'],T3_Full['Power Ø [kW]'], 'bo')
plt.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('kW', color ='r', fontsize = 20)
plt.xlabel('Windspeed m/s', color = 'r', fontsize = 20)
plt.title('T3_Full 10-minute Averaged Power Values', color = 'r', fontsize = 20)
plt.legend(['Power Ø','Power Curve'], prop={'size': 15})
plt.grid()
plt.rcParams["figure.figsize"] = [20,10]
#plt.savefig(MONTH + '/Images/Turbine_1_Power_Curve.jpg')
plt.show()

plt.plot(T4_Full['Wind Ø [m/s]'],T4_Full['Power Ø [kW]'], 'bo')
plt.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.ylabel('kW', color ='r', fontsize = 20)
plt.xlabel('Windspeed m/s', color = 'r', fontsize = 20)
plt.title('T4_Full 10-minute Averaged Power Values', color = 'r', fontsize = 20)
plt.legend(['Power Ø','Power Curve'], prop={'size': 15})
plt.grid()
plt.rcParams["figure.figsize"] = [20,10]
#plt.savefig(MONTH + '/Images/Turbine_1_Power_Curve.jpg')
plt.show()

import matplotlib.ticker as ticker

fig = plt.figure(dpi=100)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(T1_Full['Wind Ø [m/s]'],T1_Full['Power Ø [kW]'], 'bo')
#ax1.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
#ax1.set_xlabel('Windspeed m/s')
ax1.xaxis.set_tick_params(rotation=0)
ax1.xaxis.set_major_locator(ticker.MaxNLocator())
ax1.set_ylabel('Power kW')
#ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
ax1.grid(linestyle='--')
ax1.set_title('T1 Filtered 10-minute Averaged Power Values', fontsize=15)

ax2.plot(T2_Full['Wind Ø [m/s]'],T2_Full['Power Ø [kW]'], 'bo')
#ax2.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
#ax2.set_xlabel('Windspeed m/s')
ax2.xaxis.set_tick_params(rotation=0, color='b')
ax2.xaxis.set_major_locator(ticker.MaxNLocator())
ax2.set_ylabel('Power kW')
#ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
ax2.grid(linestyle='--')
ax2.set_title('T2 Filtered 10-minute Averaged Power Values', fontsize=15)

ax3.plot(T3_Full['Wind Ø [m/s]'],T3_Full['Power Ø [kW]'], 'bo')
#ax3.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
ax3.set_xlabel('Windspeed m/s')
ax3.xaxis.set_tick_params(rotation=0, color='b')
ax3.xaxis.set_major_locator(ticker.MaxNLocator())
ax3.set_ylabel('Power kW')
#ax3.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
ax3.grid(linestyle='--')
ax3.set_title('T3 Filtered 10-minute Averaged Power Values', fontsize=15)

ax4.plot(T4_Full['Wind Ø [m/s]'],T4_Full['Power Ø [kW]'], 'bo')
#ax4.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
ax4.set_xlabel('Windspeed m/s')
ax4.xaxis.set_tick_params(rotation=0, color='b')
ax4.xaxis.set_major_locator(ticker.MaxNLocator())
ax4.set_ylabel('Power kW')
#ax4.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
ax4.grid(linestyle='--')
ax4.set_title('T4 Filtered 10-minute Averaged Power Values', fontsize=15)

fig.savefig(img + '/Power_Performance_vs_Warrantied_Power_Curve.png', bbox_inches='tight')
fig.suptitle('Filtered Power Dataset', fontsize=20, color='r')
plt.show()

import matplotlib.ticker as ticker

fig = plt.figure(dpi=100)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

# Assuming max_power_value is the maximum power value in your dataset
max_power_value_filtered = 2350  # You may need to adjust this based on your data

# Normalize power values to Percentage Rated Capacity for the filtered dataset
T1_Full['Percentage Rated Capacity'] = (T1_Full['Power Ø [kW]'] / max_power_value_filtered) * 100
T2_Full['Percentage Rated Capacity'] = (T2_Full['Power Ø [kW]'] / max_power_value_filtered) * 100
T3_Full['Percentage Rated Capacity'] = (T3_Full['Power Ø [kW]'] / max_power_value_filtered) * 100
T4_Full['Percentage Rated Capacity'] = (T4_Full['Power Ø [kW]'] / max_power_value_filtered) * 100

ax1.plot(T1_Full['Wind Ø [m/s]'], T1_Full['Percentage Rated Capacity'], 'bo')
ax1.xaxis.set_tick_params(rotation=0)
ax1.xaxis.set_major_locator(ticker.MaxNLocator())
ax1.set_ylabel('Percentage Rated Capacity (%)')
ax1.grid(linestyle='--')
ax1.set_title('T1 Filtered 10-minute Averaged Power Values', fontsize=15)
ax1.set_xlim(0, 25)  # Set the x-axis limit

ax2.plot(T2_Full['Wind Ø [m/s]'], T2_Full['Percentage Rated Capacity'], 'bo')
ax2.xaxis.set_tick_params(rotation=0, color='b')
ax2.xaxis.set_major_locator(ticker.MaxNLocator())
ax2.set_ylabel('Percentage Rated Capacity (%)')
ax2.grid(linestyle='--')
ax2.set_title('T2 Filtered 10-minute Averaged Power Values', fontsize=15)
ax2.set_xlim(0, 25)  # Set the x-axis limit

ax3.plot(T3_Full['Wind Ø [m/s]'], T3_Full['Percentage Rated Capacity'], 'bo')
ax3.set_xlabel('Windspeed m/s')
ax3.xaxis.set_tick_params(rotation=0, color='b')
ax3.xaxis.set_major_locator(ticker.MaxNLocator())
ax3.set_ylabel('Percentage Rated Capacity (%)')
ax3.grid(linestyle='--')
ax3.set_title('T3 Filtered 10-minute Averaged Power Values', fontsize=15)
ax3.set_xlim(0, 25)  # Set the x-axis limit

ax4.plot(T4_Full['Wind Ø [m/s]'], T4_Full['Percentage Rated Capacity'], 'bo')
ax4.set_xlabel('Windspeed m/s')
ax4.xaxis.set_tick_params(rotation=0, color='b')
ax4.xaxis.set_major_locator(ticker.MaxNLocator())
ax4.set_ylabel('Percentage Rated Capacity (%)')
ax4.grid(linestyle='--')
ax4.set_title('T4 Filtered 10-minute Averaged Power Values', fontsize=15)
ax4.set_xlim(0, 25)  # Set the x-axis limit

fig.savefig(img + '/Filtered_Power_Performance_vs_Warrantied_Power_Curve.png', bbox_inches='tight')
fig.suptitle('Filtered Power Dataset', fontsize=20, color='r')
plt.show()


#Now I want to create small dfs to look at the binning technique a bit more
T1_F = T1_Full[['Wind Ø [m/s]', 'Power Ø [kW]']].reset_index(drop=True)
T2_F = T2_Full[['Wind Ø [m/s]', 'Power Ø [kW]']].reset_index(drop=True)
T3_F = T3_Full[['Wind Ø [m/s]', 'Power Ø [kW]']].reset_index(drop=True)
T4_F = T4_Full[['Wind Ø [m/s]', 'Power Ø [kW]']].reset_index(drop=True)

#T4_F.describe()

#T4_F

import pandas as pd
import matplotlib.pyplot as plt

# Initialize lists to store outliers and non-outliers
T1_outliers_list = []
T1_non_outliers_list = []

# Initialize lists to store corresponding wind speeds
T1_wind_speeds_outliers = []
T1_wind_speeds_non_outliers = []

# Plot box plot for each wind speed bin
for wind_speed in T1_F['Wind Ø [m/s]'].unique():
    power_data = T1_F[T1_F['Wind Ø [m/s]'] == wind_speed]['Power Ø [kW]']
    
    # Calculate Z-scores for power data in the bin
    z_scores = (power_data - power_data.mean()) / power_data.std()
    
    # Identify outliers based on Z-score threshold (e.g., 3)
    outliers = power_data[abs(z_scores) > 2.5]
    non_outliers = power_data[abs(z_scores) <= 2.5]
    
    # Append outliers and non-outliers to lists
    T1_outliers_list.append(outliers)
    T1_non_outliers_list.append(non_outliers)
    
    # Append wind speed values to corresponding lists
    T1_wind_speeds_outliers.extend([wind_speed] * len(outliers))
    T1_wind_speeds_non_outliers.extend([wind_speed] * len(non_outliers))

    # Plot outliers and non-outliers
    plt.scatter([wind_speed] * len(outliers), outliers, color='red', label='Outliers')
    plt.scatter([wind_speed] * len(non_outliers), non_outliers, color='blue', label='Non-Outliers')

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Set plot title
plt.title('Wind Speed vs. Power with Outliers and Non-Outliers', fontsize=16)

# Show the legend
#plt.legend()

# Show the plot
plt.show()

# Convert lists to DataFrames including wind speed component
T1_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T1_wind_speeds_outliers, 'Power Ø [kW]': pd.concat(T1_outliers_list).reset_index(drop=True)})
T1_non_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T1_wind_speeds_non_outliers, 'Power Ø [kW]': pd.concat(T1_non_outliers_list).reset_index(drop=True)})

# Plot non-outliers from T1
plt.plot(T1_non_outliers_df['Wind Ø [m/s]'], T1_non_outliers_df['Power Ø [kW]'], 'bo')

# Plot Power Curve (PC) from T1
plt.plot(PC['Wind (m/s)'], PC['Power (kW)'], 'r', linewidth=3)

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Show the plot
plt.show()


import matplotlib.pyplot as plt

# Initialize lists to store outliers and non-outliers
T2_outliers_list = []
T2_non_outliers_list = []

# Initialize lists to store corresponding wind speeds
T2_wind_speeds_outliers = []
T2_wind_speeds_non_outliers = []

# Plot box plot for each wind speed bin
for wind_speed in T2_F['Wind Ø [m/s]'].unique():
    power_data = T2_F[T2_F['Wind Ø [m/s]'] == wind_speed]['Power Ø [kW]']
    
    # Calculate Z-scores for power data in the bin
    z_scores = (power_data - power_data.mean()) / power_data.std()
    
    # Identify outliers based on Z-score threshold (e.g., 3)
    outliers = power_data[abs(z_scores) > 2.5]
    non_outliers = power_data[abs(z_scores) <= 2.5]
    
    # Append outliers and non-outliers to lists
    T2_outliers_list.append(outliers)
    T2_non_outliers_list.append(non_outliers)
    
    # Append wind speed values to corresponding lists
    T2_wind_speeds_outliers.extend([wind_speed] * len(outliers))
    T2_wind_speeds_non_outliers.extend([wind_speed] * len(non_outliers))

    # Plot outliers and non-outliers
    plt.scatter([wind_speed] * len(outliers), outliers, color='red', label='Outliers')
    plt.scatter([wind_speed] * len(non_outliers), non_outliers, color='blue', label='Non-Outliers')

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Set plot title
plt.title('Wind Speed vs. Power with Outliers and Non-Outliers', fontsize=16)

# Show the legend
#plt.legend()

# Show the plot
plt.show()

# Convert lists to DataFrames including wind speed component
T2_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T2_wind_speeds_outliers, 'Power Ø [kW]': pd.concat(T2_outliers_list).reset_index(drop=True)})
T2_non_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T2_wind_speeds_non_outliers, 'Power Ø [kW]': pd.concat(T2_non_outliers_list).reset_index(drop=True)})

# Plot non-outliers from T2
plt.plot(T2_non_outliers_df['Wind Ø [m/s]'], T2_non_outliers_df['Power Ø [kW]'], 'bo')

# Plot Power Curve (PC) from T2
plt.plot(PC['Wind (m/s)'], PC['Power (kW)'], 'r', linewidth=3)

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Show the plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Initialize lists to store outliers and non-outliers
T3_outliers_list = []
T3_non_outliers_list = []

# Initialize lists to store corresponding wind speeds
T3_wind_speeds_outliers = []
T3_wind_speeds_non_outliers = []

# Plot box plot for each wind speed bin
for wind_speed in T3_F['Wind Ø [m/s]'].unique():
    power_data = T3_F[T3_F['Wind Ø [m/s]'] == wind_speed]['Power Ø [kW]']
    
    # Calculate Z-scores for power data in the bin
    z_scores = (power_data - power_data.mean()) / power_data.std()
    
    # Identify outliers based on Z-score threshold (e.g., 3)
    outliers = power_data[abs(z_scores) > 2.5]
    non_outliers = power_data[abs(z_scores) <= 2.5]
    
    # Append outliers and non-outliers to lists
    T3_outliers_list.append(outliers)
    T3_non_outliers_list.append(non_outliers)
    
    # Append wind speed values to corresponding lists
    T3_wind_speeds_outliers.extend([wind_speed] * len(outliers))
    T3_wind_speeds_non_outliers.extend([wind_speed] * len(non_outliers))

    # Plot outliers and non-outliers
    plt.scatter([wind_speed] * len(outliers), outliers, color='red', label='Outliers')
    plt.scatter([wind_speed] * len(non_outliers), non_outliers, color='blue', label='Non-Outliers')

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Set plot title
plt.title('Wind Speed vs. Power with Outliers and Non-Outliers', fontsize=16)

# Show the legend
#plt.legend()

# Show the plot
plt.show()

# Convert lists to DataFrames including wind speed component
T3_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T3_wind_speeds_outliers, 'Power Ø [kW]': pd.concat(T3_outliers_list).reset_index(drop=True)})
T3_non_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T3_wind_speeds_non_outliers, 'Power Ø [kW]': pd.concat(T3_non_outliers_list).reset_index(drop=True)})

# Plot non-outliers from T3
plt.plot(T3_non_outliers_df['Wind Ø [m/s]'], T3_non_outliers_df['Power Ø [kW]'], 'bo')

# Plot Power Curve (PC) from T3
plt.plot(PC['Wind (m/s)'], PC['Power (kW)'], 'r', linewidth=3)

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Show the plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Initialize lists to store outliers and non-outliers
T4_outliers_list = []
T4_non_outliers_list = []

# Initialize lists to store corresponding wind speeds
T4_wind_speeds_outliers = []
T4_wind_speeds_non_outliers = []

# Plot box plot for each wind speed bin
for wind_speed in T4_F['Wind Ø [m/s]'].unique():
    power_data = T4_F[T4_F['Wind Ø [m/s]'] == wind_speed]['Power Ø [kW]']
    
    # Calculate Z-scores for power data in the bin
    z_scores = (power_data - power_data.mean()) / power_data.std()
    
    # Identify outliers based on Z-score threshold
    outliers = power_data[abs(z_scores) > 2.5]
    non_outliers = power_data[abs(z_scores) <= 2.5]
    
    # Append outliers and non-outliers to lists
    T4_outliers_list.append(outliers)
    T4_non_outliers_list.append(non_outliers)
    
    # Append wind speed values to corresponding lists
    T4_wind_speeds_outliers.extend([wind_speed] * len(outliers))
    T4_wind_speeds_non_outliers.extend([wind_speed] * len(non_outliers))

    # Plot outliers and non-outliers
    plt.scatter([wind_speed] * len(outliers), outliers, color='red', label='Outliers')
    plt.scatter([wind_speed] * len(non_outliers), non_outliers, color='blue', label='Non-Outliers')

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Set plot title
plt.title('T4 Wind Speed vs. Power with Outliers and Non-Outliers', fontsize=16)

# Show the legend
#plt.legend()

# Show the plot
plt.show()

# Convert lists to DataFrames including wind speed component
T4_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T4_wind_speeds_outliers, 'Power Ø [kW]': pd.concat(T4_outliers_list).reset_index(drop=True)})
T4_non_outliers_df = pd.DataFrame({'Wind Ø [m/s]': T4_wind_speeds_non_outliers, 'Power Ø [kW]': pd.concat(T4_non_outliers_list).reset_index(drop=True)})

plt.plot(T4_non_outliers_df['Wind Ø [m/s]'], T4_non_outliers_df['Power Ø [kW]'], 'bo')
plt.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)
# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

plt.show()


#T4_non_outliers_df
###  So now we have all the individual wind turbine power curves stored we will go back and build a better farm power curve.

# Concatenate all the DataFrames into one DataFrame
all_non_outliers_df = pd.concat([T1_non_outliers_df, T2_non_outliers_df, T3_non_outliers_df, T4_non_outliers_df])

# Calculate the mean power for each unique wind speed
Farm_df = all_non_outliers_df.groupby('Wind Ø [m/s]').mean().reset_index()


# Apply Gaussian filter to smoothen the power curve
Farm_smoothed_power = gaussian_filter1d(Farm_df['Power Ø [kW]'], sigma=2)

# Plot the original and smoothed power curves
#plt.plot(Farm_df['Wind Ø [m/s]'], Farm_df['Power Ø [kW]'], label='Original Power Curve', color='blue')
plt.plot(Farm_df['Wind Ø [m/s]'], Farm_smoothed_power, label='Farm Smoothed Power Curve', color='blue')

#plt.plot(PC['Wind (m/s)'],PC['Power (kW)'],'r',linewidth=3)

# Set x-axis label
plt.xlabel('Wind Speed (m/s)', fontsize=14)

# Set y-axis label
plt.ylabel('Power (kW)', fontsize=14)

# Set plot title
plt.title('Farm Power Curve with Gaussian Smoothing', fontsize=16)

# Set the maximum value of x-axis to 25 m/s
plt.xlim(0, 25)

# Add legend
plt.legend()

# Show the plot
plt.show()

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# Assuming you have Farm_df DataFrame and max_power_value variable

# Normalize power values to percentages of maximum rated capacity
Farm_df['Percentage Rated Capacity (%)'] = (Farm_df['Power Ø [kW]'] / max_power_value) * 100

# Apply Gaussian filter to smoothen the power curve
smoothed_power = gaussian_filter1d(Farm_df['Percentage Rated Capacity (%)'], sigma=2)

# Plot the original and smoothed power curves
#plt.plot(Farm_df['Wind Ø [m/s]'], Farm_df['Percentage Rated Capacity (%)'], label='Original Power Curve', color='blue')
# plt.plot(Farm_df['Wind Ø [m/s]'], smoothed_power, label='Farm Power Curve', color='blue',linewidth=3)

# # Set x-axis label
# plt.xlabel('Wind Speed (m/s)', fontsize=14)

# # Set y-axis label
# plt.ylabel('Percentage Rated Capacity (%)', fontsize=14)

# # Set plot title
# #plt.title('Farm Power Curve with Gaussian Smoothing', fontsize=16)

# # Set the maximum value of x-axis to 25 m/s
# plt.xlim(0, 25)

# # Add gridlines with dashed lines
# plt.grid(linestyle='--')

# # Set x-axis ticks at intervals of 2.5 m/s
# plt.xticks(np.arange(0, 26, 2.5))

# # Add legend
# #plt.legend()

# # Show the plot
# plt.show()

