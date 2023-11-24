#!/usr/bin/env python
# coding: utf-8

# # Introduction

# <center><img src="https://i.imgur.com/9hLRsjZ.jpg" height=400></center>
# 
# This dataset was scraped from [nextspaceflight.com](https://nextspaceflight.com/launches/past/?page=1) and includes all the space missions since the beginning of Space Race between the USA and the Soviet Union in 1957!

# ### Install Package with Country Codes

# In[2]:


get_ipython().run_line_magic('pip', 'install iso3166')


# In[3]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# ### Upgrade Plotly
# 
# Run the cell below if you are working with Google Colab.

# In[4]:


get_ipython().run_line_magic('pip', 'install --upgrade plotly')


# ### Import Statements

# In[41]:


import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# These might be helpful:
from iso3166 import countries
from datetime import datetime, timedelta


# ### Notebook Presentation

# In[6]:


pd.options.display.float_format = '{:,.2f}'.format


# ### Load the Data

# In[7]:


df_data=pd.read_csv('E:/UDEMY/Python/Space+Missions+(start)/mission_launches.csv')


# # Preliminary Data Exploration
# 
# * What is the shape of `df_data`? 
# * How many rows and columns does it have?
# * What are the column names?
# * Are there any NaN values or duplicates?

# In[8]:


print(df_data.head())


# In[9]:


df_data.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
df_data.head()


# In[10]:


df_data.shape


# ## Data Cleaning - Check for Missing Values and Duplicates
# 
# Consider removing columns containing junk data. 

# In[11]:


print(f'Any duplicates? {df_data.duplicated().values.any()}')


# In[12]:


print(f'Any NaN values among the data? {df_data.isna().values.any()}')


# In[13]:


nan_rows = df_data[df_data.isna()]
print(nan_rows.shape)
nan_rows.head()


# In[14]:


df_data_clean = df_data.dropna()
df_data_clean.shape


# In[15]:


duplicated_rows = df_data_clean[df_data_clean.duplicated()]
print(duplicated_rows.shape)
duplicated_rows.head()


# In[16]:


df_data_clean = df_data_clean.drop_duplicates()


# In[17]:


df_data_clean.shape


# ## Descriptive Statistics

# In[18]:


df_data_clean.info()


# In[77]:


df_data_clean.describe()


# In[47]:


df_data_clean.sample(5)


# # Number of Launches per Company
# 
# Create a chart that shows the number of space mission launches by organisation.

# In[49]:


company = df_data_clean.Organisation.value_counts()
company


# In[50]:


v_bar = px.bar(
        x = company.index,
        y = company.values,
        color = company.values,
        color_continuous_scale='Aggrnyl',
        title='Number of Companies')

v_bar.update_layout(xaxis_title='Organisation', 
                    coloraxis_showscale=False,
                    yaxis_title='Number of Launches')
v_bar.show()


# # Number of Active versus Retired Rockets
# 
# How many rockets are active compared to those that are decomissioned? 

# In[51]:


status = df_data_clean.Rocket_Status.value_counts()
status


# In[52]:


fig = px.pie(labels=status.index, 
             values=status.values,
             title="Number of Active versus Retired Rockets",
             names=status.index,
             hole=0.4,)

fig.update_traces(textposition='inside', textfont_size=15, textinfo='percent')

fig.show()


# # Distribution of Mission Status
# 
# How many missions were successful?
# How many missions failed?

# In[53]:


mission_status = df_data_clean.Mission_Status.value_counts()
mission_status


# In[54]:


fig = px.pie(labels=mission_status.index,
             values=mission_status.values,
             title="Mission Status",
             names=mission_status.index,
)
fig.update_traces(textposition='outside', textinfo='percent+label')

fig.show()


# # How Expensive are the Launches? 
# 
# Create a histogram and visualise the distribution. The price column is given in USD millions (careful of missing values). 

# In[55]:


df_data_clean.Price.describe()


# In[64]:


price = df_data_clean.Price.value_counts()
price


# In[26]:


plt.figure(figsize=(12, 8))
sns.histplot(data=df_data_clean,
             x=df_data_clean.Price,
             bins=30)
plt.xticks(rotation=45, fontsize=8)
plt.xlabel('Price')
plt.title('Distribution of Expences')
plt.show()


# # Use a Choropleth Map to Show the Number of Launches by Country
# 
# * Create a choropleth map using [the plotly documentation](https://plotly.com/python/choropleth-maps/)
# * Experiment with [plotly's available colours](https://plotly.com/python/builtin-colorscales/). I quite like the sequential colour `matter` on this map. 
# * You'll need to extract a `country` feature as well as change the country names that no longer exist.
# 
# Wrangle the Country Names
# 
# You'll need to use a 3 letter country code for each country. You might have to change some country names.
# 
# * Russia is the Russian Federation
# * New Mexico should be USA
# * Yellow Sea refers to China
# * Shahrud Missile Test Site should be Iran
# * Pacific Missile Range Facility should be USA
# * Barents Sea should be Russian Federation
# * Gran Canaria should be USA
# 
# 
# You can use the iso3166 package to convert the country names to Alpha3 format.

# In[27]:


df_data_clean['Location'] = df_data_clean['Location'].str.split(',').str[-1].str.strip()
df_data_clean['Location']


# In[118]:


# Define a mapping for replacements
replacement_mapping = {
    'Russia': 'Russian Federation',
    'New Mexico': 'USA',
    'Yellow Sea': 'China',
    'Shahrud Missile Test Site': 'Iran',
    'Pacific Missile Range Facility': 'USA',
    'Barents Sea': 'Russian Federation',
    'Gran Canaria': 'USA',
    'KAZ': 'Russian Federation',
}

# Replace values based on the mapping
df_data_clean['Location'] = df_data_clean['Location'].replace(replacement_mapping)


# In[120]:


df_data_clean.head()


# In[30]:


df_data_clean['Location'] = df_data_clean['Location'].apply(lambda x: countries.get(x).alpha3 if countries.get(x) else None)


# In[31]:


df_data_clean.head()


# In[138]:


df_data_clean['Location']


# In[32]:


location = df_data_clean.Location.value_counts()
location


# In[158]:


fig = px.choropleth(location, geojson=location.index, locations=location.index, color='count',
                           color_continuous_scale="Viridis",
                           range_color=(0, 490),
                           labels={'lbc':'Launches by Country'}
                          )
fig.update_geos(projection_type="natural earth")
fig.update_layout(height=500, margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# # Use a Choropleth Map to Show the Number of Failures by Country
# 

# In[177]:


faliure_counts = df_data_clean[df_data_clean['Mission_Status'] == 'Failure'].groupby('Location').size().reset_index(name='Count')
faliure_counts


# In[179]:


faliure_counts.shape


# In[183]:


failure_counts = df_data_clean[df_data_clean['Mission_Status'] == 'Failure'].groupby('Location').size().reset_index(name='Count')

fig = px.choropleth(failure_counts, geojson=failure_counts['Location'], locations=failure_counts['Location'], color='Count',
                    color_continuous_scale="Viridis",
                    range_color=(0, failure_counts['Count'].max()),
                    labels={'Count': 'Launches by Country'}
                    )
fig.update_geos(projection_type="natural earth")
fig.update_layout(height=500, margin={"r": 0, "t": 0, "l": 0, "b": 0})
fig.show()


# # Create a Plotly Sunburst Chart of the countries, organisations, and mission status. 

# In[184]:


df_data_clean.head()


# In[189]:


sunburst_chart = df_data_clean.groupby(by=['Location', 
                                       'Organisation'], as_index=False).agg({'Mission_Status': pd.Series.count})

sunburst_chart = sunburst_chart.sort_values('Mission_Status', ascending=False)
sunburst_chart


# In[191]:


burst = px.sunburst(sunburst_chart, 
                    path=['Location', 'Organisation'], 
                    values='Mission_Status',
                    title='Countries, Organisations, and Mission Status?',
                   )

burst.update_layout(xaxis_title='Mission_Status', 
                    yaxis_title='Location',
                    coloraxis_showscale=False)

burst.show()


# # Analyse the Total Amount of Money Spent by Organisation on Space Missions

# In[202]:


df_data_clean['Price'] = df_data_clean['Price'].replace(',', '', regex=True).astype(float)


# In[205]:


total_spending = df_data_clean.groupby('Organisation')['Price'].sum().reset_index()
total_spending


# In[208]:


fig = px.bar(total_spending, x='Organisation', y='Price',
             labels={'Price': 'Total Money Spent'},
             title='Total Money Spent by Organization')

fig.show()


# # Analyse the Amount of Money Spent by Organisation per Launch

# In[209]:


df_data_clean.head()


# In[213]:


launch_counts = df_data_clean['Organisation'].value_counts().reset_index()
launch_counts.columns = ['Organisation', 'Launch_Count']
launch_counts


# In[214]:


result_df = pd.merge(total_spending, launch_counts, on='Organisation')
result_df


# In[217]:


result_df['Money_Spent_Per_Launch'] = result_df['Price'] / result_df['Launch_Count']
result_df


# # Chart the Number of Launches per Year

# In[263]:


df_data_clean.dtypes


# In[266]:


df_data_clean.Date = pd.to_datetime(df_data_clean['Date'])


# In[267]:


df_data_clean['year'] = df_data_clean['Date'].dt.year

# Group by year and count the number of launches
launch_counts_per_year = df_data_clean.groupby('year').size().reset_index(name='Number_of_Launches')


# In[268]:


fig = px.bar(launch_counts_per_year, x='year', y='Number_of_Launches',
             labels={'Number_of_Launches': 'Number of Launches'},
             title='Number of Launches per Year')

# Show the chart
fig.show()


# # Chart the Number of Launches Month-on-Month until the Present
# 
# Which month has seen the highest number of launches in all time? Superimpose a rolling average on the month on month time series chart. 

# In[38]:


# Assuming df_data_clean is your DataFrame
df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'], errors='coerce')

# Drop rows with invalid dates
df_data_clean = df_data_clean.dropna(subset=['Date'])

# Extract year-month from the 'Date' column
df_data_clean['year_month'] = df_data_clean['Date'].dt.to_period('M')

# Group by year-month and count the number of launches
launches_monthly = df_data_clean.groupby('year_month').size().reset_index(name='Number_of_Launches')



# In[39]:


df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Extract month from the 'date' column
df_data_clean['month'] = df_data_clean['Date'].dt.strftime('%B')  # Full month name


# Extract month from the 'date' column
df_data_clean['month'] = df_data_clean['Date'].dt.to_period('M')

# Group by month and count the number of launches
launches_monthly = df_data_clean.groupby('month').size().reset_index(name='Number_of_Launches')

# Plot the results using a bar plot
plt.figure(figsize=(10, 6))
plt.bar(launches_monthly['month'].astype(str), launches_monthly['Number_of_Launches'])
plt.title('Number of Launches per Month (All Years)')
plt.xlabel('Month', fontsize=8)
plt.ylabel('Number of Launches')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # Launches per Month: Which months are most popular and least popular for launches?
# 
# Some months have better weather than others. Which time of year seems to be best for space missions?

# In[ ]:





# In[284]:


df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Extract month from the 'date' column
df_data_clean['month'] = df_data_clean['Date'].dt.strftime('%B')  # Full month name

# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Convert 'month' column to categorical with the specified order
launches_monthly['month'] = pd.Categorical(launches_monthly['month'], categories=month_order, ordered=True)

# Sort the DataFrame by the categorical 'month' column
launches_monthly = launches_monthly.sort_values('month')

# Plot the results using a bar plot
plt.figure(figsize=(10, 6))
plt.bar(launches_monthly['month'], launches_monthly['Number_of_Launches'])
plt.title('Number of Launches per Month (All Years)')
plt.xlabel('Month')
plt.ylabel('Number of Launches')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# # How has the Launch Price varied Over Time? 
# 
# Create a line chart that shows the average price of rocket launches over time. 

# In[291]:


# Set the Seaborn color palette to Viridis
sns.set_palette("viridis")

# Fit a curve (polynomial regression) to the data
degree = 3  # Adjust the degree of the polynomial as needed
coefficients = np.polyfit(df_data_clean['Date'].view(np.int64), df_data_clean['Price'], degree)
poly_fit = np.poly1d(coefficients)

# Plot the launch price variation over time with a scatter plot and a fitted curve
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_data_clean['Date'], y=df_data_clean['Price'], label='Actual Data')
sns.lineplot(x=df_data_clean['Date'], y=poly_fit(df_data_clean['Date'].view(np.int64)), label=f'Polynomial Fit (Degree {degree})', color='red')
plt.title('Launch Price Variation Over Time with Scatter Plot and Curve Fit')
plt.xlabel('Date')
plt.ylabel('Launch Price')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1500)
plt.tight_layout()
plt.show()


# In[54]:


# Plot the launch price variation over time
plt.figure(figsize=(10, 6))
plt.plot(df_data_clean['Date'], df_data_clean['Price'])
plt.title('Launch Price Variation Over Time')
plt.xlabel('Date')
plt.ylabel('Launch Price', fontsize=4)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 55)
plt.tight_layout()
plt.show()


# # Chart the Number of Launches over Time by the Top 10 Organisations. 
# 
# How has the dominance of launches changed over time between the different players? 

# In[75]:


# Convert the 'Date' column to datetime format
df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Extract the top 10 organizations
top_10_organizations = df_data_clean['Organisation'].value_counts().nlargest(10).index

# Filter the DataFrame for the top 10 organizations
df_top_10 = df_data_clean[df_data_clean['Organisation'].isin(top_10_organizations)].copy()

# Extract the year from the 'Date' column
df_top_10['Year'] = df_top_10['Date'].dt.year

# Plot the number of launches over time
plt.figure(figsize=(16, 8))
sns.countplot(x='Year', hue='Organisation', data=df_top_10, palette='viridis', dodge=True, saturation=0.75, linewidth=10)
plt.title('Number of Launches Over Time by Top 10 Organizations')
plt.xlabel('Year', fontsize='2')
plt.ylabel('Number of Launches')
plt.xticks(rotation=45)
plt.legend(title='Organization', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 30) 
plt.show()


# # Cold War Space Race: USA vs USSR
# 
# The cold war lasted from the start of the dataset up until 1991. 

# In[121]:


# Define a mapping for replacements
replacement_mapping = {
    'KAZ': 'RUS',
}

# Replace values based on the mapping
df_data_clean['Location'] = df_data_clean['Location'].replace(replacement_mapping)


# In[122]:


# Specify the desired locations
desired_locations = ['RUS', 'USA']

# Convert the 'Date' column to datetime format
df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Filter the DataFrame to include only rows with 'Location' in the desired locations
filtered_data = df_data_clean[df_data_clean['Location'].isin(desired_locations)]

# Filter the DataFrame to include only rows with a year less than or equal to 1991
filtered_data = filtered_data[filtered_data['Date'].dt.year <= 1991]

filtered_data


# In[123]:


# Extract the year from the 'Date' column
filtered_data['Year'] = filtered_data['Date'].dt.year

# Plot the number of launches over time
plt.figure(figsize=(16, 8))
sns.countplot(x='Year', hue='Location', data=filtered_data, palette='viridis', dodge=True, saturation=0.75, linewidth=10)
plt.title('Space Race: USA vs USSR')
plt.xlabel('Year', fontsize='2')
plt.ylabel('Number of Launches')
plt.xticks(rotation=45)
plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 11) 
plt.show()


# ## Create a Plotly Pie Chart comparing the total number of launches of the USSR and the USA
# 
# Hint: Remember to include former Soviet Republics like Kazakhstan when analysing the total number of launches. 

# In[124]:


# Specify the desired locations
desired_locations = ['USA', 'RUS']

# Filter the DataFrame to include only rows with 'Location' in the desired locations
filtered_data = df_data_clean[df_data_clean['Location'].isin(desired_locations)]

# Group by 'Location' and count the number of launches
launch_count = filtered_data['Location'].value_counts()

# Create a Pie Chart using Plotly
fig = px.pie(launch_count, values=launch_count.values, names=launch_count.index, title='Total Number of Launches (USA vs RUS)')
fig.show()


# ## Create a Chart that Shows the Total Number of Launches Year-On-Year by the Two Superpowers

# In[125]:


# Specify the desired locations
desired_locations = ['RUS', 'USA']

# Convert the 'Date' column to datetime format
df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Filter the DataFrame to include only rows with 'Location' in the desired locations
total_number = df_data_clean[df_data_clean['Location'].isin(desired_locations)]

total_number


# In[128]:


# Extract the year from the 'Date' column
total_number['Year'] = total_number['Date'].dt.year

# Plot the number of launches over time
plt.figure(figsize=(16, 8))
sns.countplot(x='Year', hue='Location', data=total_number, palette='viridis', dodge=True, saturation=0.75, linewidth=10)
plt.title('Space Race: USA vs USSR')
plt.xlabel('Year', fontsize='2')
plt.ylabel('Number of Launches')
plt.xticks(rotation=45)
plt.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 30) 
plt.show()


# ## Chart the Total Number of Mission Failures Year on Year.

# In[130]:


# Specify the desired locations
desired_status = ['Failure']

# Convert the 'Date' column to datetime format
df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Filter the DataFrame to include only rows with 'Location' in the desired locations
faliur_counts = df_data_clean[df_data_clean['Mission_Status'].isin(desired_status)]


faliur_counts


# In[134]:


# Extract the year from the 'Date' column
faliur_counts['Year'] = faliur_counts['Date'].dt.year

# Plot the number of launches over time
plt.figure(figsize=(16, 8))
sns.countplot(x='Year', hue='Location', data=faliur_counts, palette='viridis', dodge=True, saturation=0.75, linewidth=10)
plt.title('Faliur Counts')
plt.xlabel('Year', fontsize='2')
plt.ylabel('Number of Faliur')
plt.xticks(rotation=45)
plt.legend(title='Faliur', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim(0, 2.5) 
plt.show()


# ## Chart the Percentage of Failures over Time
# 
# Did failures go up or down over time? Did the countries get better at minimising risk and improving their chances of success over time? 

# In[135]:


df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])
df_data_clean['Failure'] = df_data_clean['Mission_Status'] != 'Success'


# In[136]:


failure_percentage = df_data_clean.groupby(df_data_clean['Date'].dt.to_period("M"))['Failure'].mean() * 100


# In[137]:


# Plot the percentage of failures over time
plt.figure(figsize=(12, 6))
sns.lineplot(x=failure_percentage.index.astype(str), y=failure_percentage.values, marker='o')

# Add title and labels
plt.title('Percentage of Failures Over Time')
plt.xlabel('Date')
plt.ylabel('Failure Percentage')

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# # For Every Year Show which Country was in the Lead in terms of Total Number of Launches up to and including including 2020)
# 
# Do the results change if we only look at the number of successful launches? 

# In[138]:


# Convert 'Date' to datetime
df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Extract the year from the 'Date' column
df_data_clean['Year'] = df_data_clean['Date'].dt.year

# Count the number of launches for each country and year
launch_counts = df_data_clean.groupby(['Year', 'Location']).size().reset_index(name='LaunchCount')

# Find the country in the lead for each year
lead_country_each_year = launch_counts.loc[launch_counts.groupby('Year')['LaunchCount'].idxmax()]

# Map ISO3166 codes
lead_country_each_year['Location_ISO3166'] = lead_country_each_year['Location'].apply(lambda x: countries.get(x).alpha3 if countries.get(x) else None)

# Display the result
print(lead_country_each_year[['Year', 'Location_ISO3166', 'LaunchCount']])


# In[140]:


# Plotting the information
plt.figure(figsize=(12, 8))
sns.barplot(x='Year', y='LaunchCount', hue='Location_ISO3166', data=lead_country_each_year)
plt.title('Country with the Most Launches Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Launches')
plt.legend(title='Country', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)

# Show the plot
plt.show()


# # Create a Year-on-Year Chart Showing the Organisation Doing the Most Number of Launches
# 
# Which organisation was dominant in the 1970s and 1980s? Which organisation was dominant in 2018, 2019 and 2020? 

# In[142]:


# Convert 'Date' to datetime
df_data_clean['Date'] = pd.to_datetime(df_data_clean['Date'])

# Extract the year from the 'Date' column
df_data_clean['Year'] = df_data_clean['Date'].dt.year

# Count the number of launches for each organization and year
launch_counts = df_data_clean.groupby(['Year', 'Organisation']).size().reset_index(name='LaunchCount')

# Find the organization with the most launches for each year
most_launches_each_year = launch_counts.loc[launch_counts.groupby('Year')['LaunchCount'].idxmax()]

# Plotting the information
plt.figure(figsize=(12, 8))
sns.barplot(x='Year', y='LaunchCount', hue='Organisation', data=most_launches_each_year)
plt.title('Organization with the Most Launches Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Launches')
plt.legend(title='Organization', bbox_to_anchor=(1, 1))
plt.xticks(rotation=45)

# Show the plot
plt.show()

