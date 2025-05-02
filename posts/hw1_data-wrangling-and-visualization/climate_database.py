# importing necessary packages
import sqlite3
import numpy as np
import pandas as pd
from plotly import express as px
from sklearn.linear_model import LinearRegression

def query_climate_database(db_file, country, year_begin, year_end, month) :
    """ Extracts climate data from a specified database based on the provided country, 
        year range, and month; and returns extracted data in a Pandas dataframe

    Args:
        db_file (string): file name for the database
        country (string): name of the country for which data should be returned
        year_begin (integer): the earliest year for which should be returned
        year_end (integer): the latest years for which to should returned
        month (integer): the month of the year for which should be returned

    Returns:
        df (Pandas dataframe): a dataframe of temperature readings of inputted country according to 
            inputted year_begin, inputted year_end, and inputted month of the year. The
            resulting dataframe contains the columns `NAME`, 'LATITUDE', 'LONGITUDE`, 
            `Country`, `Year`, `Month`, `Temp`.
    """

    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("""
                    SELECT s.*, c.Name AS Country, t.*
                    FROM stations s
                    INNER JOIN countries c
                    ON c."FIPS 10-4" = SUBSTR(s.ID, 1, 2)
                    INNER JOIN temperatures t
                    ON t.ID = s.ID
                    WHERE c.Name = ? AND t.Year >= ? AND t.Year <= ? AND t.Month = ?   
                    ORDER BY NAME                            
                    """, conn, params=(country, year_begin, year_end, month))
        
        # returns dataframe in order of columns mentioned before
    return df[["NAME", "LATITUDE", "LONGITUDE", "Country", "Year", "Month", "Temp"]]    


def coef(data_group):
    """ fits data to a linear regression model and outputs the first coefficient of the fitted model

    Args:
        data_group (Pandas dataframe): the data for which to fit to a model
    
    Returns:
        
    """
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]


def temperature_coefficient_plot(db_file, country, year_begin, year_end, month, min_obs, **kwargs) : 
    """creates a geographic scatter plot where each point displays the first coefficent
            of the linear regression model fitted for each station's temperature in given year range

    Args:
        db_file (string): file name for the database
        country (string): name of the country for which data should be returned
        year_begin (integer): the earliest year for which should be returned
        year_end (integer): the latest years for which to should returned
        month (integer): the month of the year for which should be returned
        min_obs (integer): the minimum required number of years of data for any given station
        **kwargs (optional): keyword arguments for the geographic scatter plot
    
    Returns:
        an interactive geographic scatterplot
    """

    df = query_climate_database(db_file, country, year_begin, year_end, month)

    # Filter out stations with observations less than min_obs
    df['ObsCount'] = df.groupby('NAME')['NAME'].transform('count')
    df = df[df['ObsCount'] >= min_obs]

    # Fitting each station data to a linear regression model
    #   and put the first coefficient of the fitted models into a Pandas dataframe
    coefs = df.groupby(["NAME", "Month"]).apply(coef).round(4).reset_index()

    # Adding columns for latitude and longitude for each station in coefs
    lat_lon = df[['NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates('NAME')
    coefs = coefs.merge(lat_lon, how='left', left_on='NAME', right_on='NAME')
    
    # Creating dictionary to match month to month name (for title of plot)
    month_dict={1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August", 9: "September",
            10: "October", 11: "November", 12: "December"}
    
    # Preparing the plot
    fig = px.scatter_mapbox(coefs,
                            lat="LATITUDE",
                            lon="LONGITUDE",
                            hover_name="NAME",
                            color=0,
                            title = f"Estimates of yearly increase in termperature in {month_dict[month]} <br>for stations in {country}, years {year_begin} - {year_end}",
                            **kwargs)
    
    # Updating margin and colorbar range
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, 
                      coloraxis=dict(cmax=0.1, cmin=-0.1), # to make colorbar range from -0.1 to 0.1
                      coloraxis_colorbar=dict(title='Estimated Yearly<br>Increase (°C)<br>')
                      )
    
    return fig


def query_country_climate_database(db_file, country, year_begin, year_end) :
    """ Extracts climate data from a specified database based on the provided
        year range, and month; and returns extracted data in a Pandas dataframe

    Args:
        db_file (string): file name for the database
        country (string): name of the country for which data should be returned
        year_begin (integer): the earliest year for which should be returned
        year_end (integer): the latest years for which to should returned

    Returns:
        df (Pandas dataframe): a dataframe of temperature readings of inputted country according to 
            inputted year_begin, inputted year_end, and inputted month of the year. The
            resulting dataframe contains the columns `NAME`, 'LATITUDE', 'LONGITUDE`, 
            `Year`, `Month`, `Temp`.
    """

    with sqlite3.connect(db_file) as conn:
        df = pd.read_sql_query("""
                    SELECT s.*, c.Name AS Country, t.*
                    FROM stations s
                    INNER JOIN countries c
                    ON c."FIPS 10-4" = SUBSTR(s.ID, 1, 2)
                    INNER JOIN temperatures t
                    ON t.ID = s.ID
                    WHERE c.Name = ? AND t.Year >= ? AND t.Year <= ?  
                    ORDER BY NAME                            
                    """, conn, params=(country, year_begin, year_end))
    return df[["NAME", "LATITUDE", "LONGITUDE", "Year", "Month", "Temp"]]


def yearly_avg_temp_heatmap(db_file, country, year_begin, year_end, **kwargs) : 
    """Creates a heatmap that visualizes the distribution and density of 
        temperature records across different years within a specified country. 
        Each cell in the heatmap indicates the count of stations that fall 
        within a particular 
temperature range for each year range.


    Args:
        db_file (string): file name for the database
        country (string): name of the country for which data should be returned
        year_begin (integer): the earliest year for which should be returned
        year_end (integer): the latest years for which to should returned
        **kwargs (optional): keyword arguments for the heatmap

    Returns:
        an interactive heatmap
    """

    # extracting desired dataframe
    df = query_country_climate_database(db_file, country, year_begin, year_end)

    # evaluating yearly average temperature for each station
    #   then aggregate dataframe to have each row represent one station for each year
    result_df = df.groupby(['NAME', 'Year']).aggregate({
                                                    'LATITUDE': 'first',
                                                    'LONGITUDE': 'first',
                                                    'Temp': 'mean'  # Calculating yearly avg temp
                                                 }).reset_index()   #   per station

    # creating heatmap
    fig = px.density_heatmap(result_df,
                    x = "Year",
                    y = "Temp",
                    range_x = [year_begin, year_end],
                    range_y = [np.mean(result_df["Temp"]) - min(result_df["Temp"]), 
                               max(result_df["Temp"])],
                    labels = {"Temp":"Average Temperature (°C)"},
                    title = f"Yearly Average Temperature of {country}<br>from Year {year_begin} - {year_end}",
                    **kwargs
                    )

    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0},
                      coloraxis=dict(cmax=100, cmin=0))
    return fig


def monthly_avg_temp_lineplots(db_file, country, year_begin, year_end, **kwargs):
    # extracting desired dataframe
    df = query_country_climate_database(db_file, country, year_begin, year_end)

    # evaluating monthly averages across all stations each year
    monthly_avg_temp = df.groupby(['Year', 'Month']).agg({'Temp': 'mean'}).reset_index()
    fig = px.line(
                    monthly_avg_temp,
                    x='Year',
                    y='Temp',
                    facet_col='Month',        # creates a separate facet for each month
                    facet_col_wrap=4,         # wraps the facets after 4 columns
                    title=f'{country} Monthly Average Temperature Trends From {year_begin} - {year_end}',
                    labels={'Temp': 'Avg Temp (°C)', 'Year': 'Year'},
                    **kwargs
                    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Avg Temp (°C)'
    )

    fig.update_xaxes(tickangle=45)  # rotates x-labels to 45 degrees

    return fig

