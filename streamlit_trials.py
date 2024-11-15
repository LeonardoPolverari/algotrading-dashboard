import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import seaborn as sns
import backtesting as bck
from plotly import tools
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
import datetime as dt
from pandas_datareader.yahoo.fx import YahooFXReader
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.lib import SignalStrategy, TrailingStrategy
from backtesting import Backtest
from scipy.signal import argrelextrema


#################### FUNCTIONS ####################
# check column types
def check_data_types(df):
    d_types = {}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            d_types[col] = "datetime"
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif pd.api.types.is_numeric_dtype(df[col]):
            d_types[col] = "numeric"
            df[col] = pd.to_numeric(df[col], errors='coerce')
        elif pd.api.types.is_object_dtype(df[col]):
            d_types[col] = "string"
            df[col] = df[col].astype(str)
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(bool)
            d_types[col] = "boolean"
        else:
            d_types[col] = "other"

    return d_types


# define moving average
def SMA(values, n):
    return pd.Series(values).rolling(n).mean() 

# define upper Bollinger band
def STD_upper(values, n, n_std): # definire funzione
    return pd.Series(values).rolling(n).mean() + n_std * pd.Series(values).rolling(n).std() 

# define lower Bollinger band
def STD_lower(values, n, n_std): 
    return pd.Series(values).rolling(n).mean() - n_std * pd.Series(values).rolling(n).std() 

# define support
def support(values, n):
    df = pd.DataFrame(values, columns=['Close']) # used dataframe to store values because of the use of argrelextrema
    df['support'] = df.iloc[argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]['Close']
    df['support'] = df['support'].fillna(method='ffill') # argrlextrema returns NaN values, so fill them with the previous value
    return df['support'].values

# define resistance
def resistance(values, n):
    df = pd.DataFrame(values, columns=['Close'])
    df['resistance'] = df.iloc[argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]['Close']
    df['resistance'] = df['resistance'].fillna(method='ffill')
    return df['resistance'].values

#################### CLASSES ####################
# define Bollinger-SupportResistance strategy
class BollingerSrMean(TrailingStrategy): # child class of TrailingStrategy (to add trailing stop loss)

    # parameters are already optimized
    horizon = 30 # 20
    n_std_upper = 3 # 2.4 or full train 2.6
    n_std_lower = 2.8 # 2.2 or full train 2.6
    stop_loss = 7
    n = 10
    # or training on last two years before the test set:
    # horizon=30,stop_loss=7,n_std_upper=2.9999999999999996,n_std_lower=2.8,n=10)

    def init(self):

      super().init() # call parent class (TrailingStrategy)
      self.sma = self.I(SMA, self.data.Close, self.horizon) # simple moving average
      self.upper = self.I(STD_upper, self.data.Close, self.horizon, self.n_std_upper) # upper band
      self.lower = self.I(STD_lower, self.data.Close, self.horizon, self.n_std_lower) # lower band
      self.support = self.I(support, self.data.Close, self.n) # support
      self.resistance = self.I(resistance, self.data.Close, self.n) # resistance   
      self.set_trailing_sl(self.stop_loss) # stop loss


    def next(self): 
      super().next() # call parent class (TrailingStrategy)
      
      if crossover(self.lower, self.data.Close) and self.data.Close[-1] >= self.support[-1]: # if price close falls below the lower band and the close price is above the support...
        self.position.close() # close position
        self.buy() # buy

      elif crossover(self.data.Close, self.upper) and self.data.Close[-1] <= self.resistance[-1]: # if price close rises above the upper band and the close price is below the resistance...
        self.position.close() # close position
        self.sell() # sell

######### PAGES ##########
def side_bar_labels(option):
    if option == 'Visualization':
        return '# Visualization 📊'
    elif option == 'Upload Data':
        return '# Upload Data 📤'
    elif option == 'Main Page':
        return '# Strategy 🧮'
# Sidebar for page navigation
page = st.sidebar.radio("Navigation", ["Upload Data", "Visualization", "Main Page"], format_func=side_bar_labels)

######### UPLOAD DATA ##########


if page == "Upload Data":
    st.title("Upload your file")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'txt'])
    if uploaded_file is not None:
        st.write("File uploaded successfully!")

        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            d_types = check_data_types(st.session_state['data'])
            st.session_state['data'] = df
            st.session_state['d_types'] = d_types
            st.write("## Data preview:")
            st.dataframe(df.head())

        except Exception as e:
            st.warning("\U000026A0 Warning: Please check your input!")
            st.write(e)

    if 'data' in st.session_state:
        with st.expander("Columns Details", expanded=False):
            
          # Iterate over the dictionary
          for col, dtype in st.session_state['d_types'].items():
              # Select new data type
              new_dtype = st.selectbox(
                  col,
                  options=["numeric", "string", "datetime", "boolean", "other"],
                  index=["numeric", "string", "datetime", "boolean", "other"].index(str(dtype).split('.')[0]),  # Set default index based on current dtype
                  key=col
              )
              
              # Change the data type if a new one is selected
              if new_dtype != 'none':
                  try:
                      if new_dtype == 'numeric':
                          st.session_state['data'][col] = pd.to_numeric(st.session_state['data'][col])
                      elif new_dtype == 'string':
                          st.session_state['data'][col] = st.session_state['data'][col].astype(str)
                      elif new_dtype == 'datetime':
                          st.session_state['data'][col] = pd.to_datetime(st.session_state['data'][col])
                      elif new_dtype == 'boolean':
                          st.session_state['data'][col] = st.session_state['data'][col].astype(bool)

                  except Exception as e:
                      st.error(f"Error changing {col} to {new_dtype}: {e}")

          # Display the modified DataFrame
          st.write("Modified DataFrame:")
          st.dataframe(st.session_state['data'])
          


######### VISUALIZATION ########## 


elif page == "Visualization":
    st.title("Visualization")
    
    # Sample DataFrame
    if 'data' not in st.session_state:
        st.warning("\U000026A0 Warning: Please upload a file first!")
    else:
        df = st.session_state['data']
        # Create two selectboxes to simulate drag-and-drop for x and y axes
        x_axis = st.multiselect("Select X-axis", df.columns, key='x_axis', max_selections=1)
        y_axis = st.multiselect("Select Y-axis (you can choose multiple)", df.columns, key='y_axis')

        if x_axis:
            st.write(f"Select the {x_axis[0]} range for the plot")
            
            # Use columns to place the date inputs in the same row
            min_val_col, max_val_col = st.columns(2)


            df[x_axis[0]].dtype #== np.datetime64

            # handle datetime x axis
            if x_axis[0] in df.select_dtypes(include = ['datetimetz', 'datetime']):
        
                with min_val_col:
                    min_x = st.date_input("Select start date", min(df[x_axis[0]]), min_value=min(df[x_axis[0]]), max_value=max(df[x_axis[0]]))
                with max_val_col:
                    max_x = st.date_input("Select end date", max(df[x_axis[0]]), min_value=min(df[x_axis[0]]), max_value=max(df[x_axis[0]]))

                min_x = pd.to_datetime(min_x)
                max_x = pd.to_datetime(max_x)
            
            # handle object dtypes
            elif x_axis[0] not in df.select_dtypes('number').columns:
                st.warning("\U000026A0 Warning: Please make sure the X axis is a number or a date!")
    
            # handle numeric data
            else:
                with min_val_col:
                    min_x = st.number_input(f"Select min {x_axis[0]} value", min_value=min(df[x_axis[0]]), max_value=max(df[x_axis[0]]), value = 'min')
                with max_val_col:
                    max_x = st.number_input(f"Select max {x_axis[0]} value", min_value=min(df[x_axis[0]]), max_value=max(df[x_axis[0]]), value = df[x_axis[0]].max())


        # Add a button to trigger the plot creation
        graph_type_col, generate_graph_col = st.columns([2, 1])

        # graph type button
        with graph_type_col:
            graph_type = st.multiselect('graph type', ['line', 'scatter', 'hist', 'candle'], key = 'graph_type', max_selections=1, default='line')

        if st.session_state['graph_type'] and st.session_state['graph_type'][0] == 'candle':
            st.write('Please specify the names of the columns containing the following values:')
            
            # Create a dictionary to hold the user-selected column names
            candle_columns = {}
            
            # List of required column names
            candle_stats = ['Date', 'Open', 'High', 'Low', 'Close']
        
            # Loop through each statistic and create a select box for it
            for stat in candle_stats:
                # Set the default index to the first occurrence of the stat in the columns list
                default_index = df.columns.tolist().index(stat) if stat in df.columns.tolist() else 0
                
                candle_columns[stat] = st.selectbox(
                    f'Select the column for {stat}', 
                    options=df.columns.tolist(),  # Provide available columns from the DataFrame
                    index=default_index,          # Set the default index based on the statistic
                    key=stat
                )
        
        # generator button
        with generate_graph_col:
            # Inject custom CSS to add padding to the button
            st.markdown("""
                <style>
                .stButton > button {
                    padding-top: 22px;
                    padding-bottom: 22px;
                }
                </style>
                """, unsafe_allow_html=True)
            
            plot_button = st.button("Create Plot")

        # Display a plot based on the user's selections after the button is clicked
        if (plot_button and x_axis and y_axis) or (plot_button and st.session_state['graph_type'][0] == 'candle'):

            if x_axis:
                plot_df = df[(df[x_axis[0]] >= min_x) & (df[x_axis[0]] <= max_x)].copy()

            # Create a plot with multiple y-values
            fig = go.Figure()

            # handle line graphs
            if st.session_state['graph_type'][0] == 'line':
                for y_col in y_axis:
                    fig.add_trace(go.Scatter(x=plot_df[x_axis[0]], y=plot_df[y_col], mode='lines', name=y_col))

            # handle line graphs
            elif st.session_state['graph_type'][0] == 'scatter':
                for y_col in y_axis:
                    fig.add_trace(go.Scatter(x=plot_df[x_axis[0]], y=plot_df[y_col], mode='markers', name=y_col))

            elif st.session_state['graph_type'][0] == 'hist':
                st.write('notice that hist will only consider the columns specified for the y axis!')
                for y_col in y_axis:
                    fig.add_trace(go.Histogram(
                        x=plot_df[y_col], 
                        name=y_col,
                        opacity=0.75,  # Add transparency for overlapping bars
                        marker=dict(
                            line=dict(width=1, color='white')  # Thin white lines to separate bars
                        ),
                    ))

            elif st.session_state['graph_type'][0] == 'candle':
                try:
                    fig.add_trace(go.Candlestick(x = plot_df[st.session_state['Date']],
                                                high = plot_df[st.session_state['High']],
                                                low = plot_df[st.session_state['Low']],
                                                open = plot_df[st.session_state['Open']],
                                                close = plot_df[st.session_state['Close']]))
                except:
                    st.warning('Please make sure the columns names are correctly specified and their type is correct (e.g. Date must be datetime)')

            # Add sliding window (range slider) to x-axis and enable box zooming
            fig.update_layout(
                # title=f"{st.session_state['graph_type'][0]} plotting {', '.join(y_axis)} on {x_axis[0]}",
                # xaxis_title=x_axis[0],
                # yaxis_title=', '.join(y_axis),
                bargap=0.1, # Gap between bars
                
            #     # Configure x-axis to have both range slider and default zoom functionality
                xaxis=dict(
                    rangeslider=dict(
                        visible=True,  # Enable the range slider
                    ),
                    rangeselector=dict(  # Optional: Range selector buttons
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(step="all")
                        ])
                    )
                ),
                
                # Enable regular plotly zooming (square selection)
                dragmode="zoom",  # This ensures the default dragmode is set to zoom
                plot_bgcolor='#f0f0f0',  # Optional: Light gray background for a Seaborn-like look
                paper_bgcolor='#f0f0f0',  # Optional: Light gray paper background
            )

            # Show the figure in Streamlit
            st.plotly_chart(fig)

            
          



######### MAIN PAGE ##########



elif page == "Main Page":
    st.title("Main Page")
    st.write("This is where the main functionality would go.")
        # Sample DataFrame
    if 'data' not in st.session_state:
        st.warning("\U000026A0 Warning: Please upload a file first!")
    else:
        df = st.session_state['data']
        # Create selectbox for indicators
        indicators = ['bollinger', 'support', 'resistance', 'sma']
        strategy = st.multiselect("Select Strategy", indicators, key='indicators', max_selections=4)
