# fitlering out VAE suggestions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import streamlit as st
import plotly.express as px

import pandas as pd
import plotly.graph_objects as go
import sys

def generate_usage(suggestions, quantiles=[0.30, 0.70], usage_threshold=0,
                  content_type='VAE', usage_level='mini_brick_code'):
    print('Generating Usage for Given suggestions')
    print('xxx----------------------------------------------------xxx')

    # 1) Filter on content_type if given
    if content_type in ('VAE', 'iDetail'):
        suggestions = suggestions[suggestions.content_type == content_type]
        flag = content_type.lower()
    else:
        # no filtering, count both
        flag = None

    # 2) Clean invalid codes and flags
    suggestions = suggestions[
        suggestions[usage_level].notna() &
        (suggestions[usage_level] != '--') &
        (suggestions.az_generated_suggestion_flag == 'Y')
    ]

    print('The number of rows are ', suggestions.shape)
    print('The number of bricks are ', suggestions[usage_level].nunique())
    print('The number of unique suggestion_pk are ', suggestions.suggestion_pk.nunique())

    # 3) Create a unified accepted_flag boolean
    if flag:
        col = f"{flag}_suggestion_transaction_flag"
        suggestions['accepted_flag'] = (suggestions[col] == 'Y')
    else:
        suggestions['accepted_flag'] = (
            (suggestions['vae_suggestion_transaction_flag'] == 'Y') |
            (suggestions['idetail_suggestion_transaction_flag'] == 'Y')
        )

    # 4) Aggregate overall usage summary
    result = (
        suggestions
        .groupby(usage_level)
        .agg(
            total_suggestions_sent       = ('suggestion_pk', 'nunique'),
            total_suggestions_sent_count = ('suggestion_pk', 'count'),
            total_suggestions_accepted   = ('accepted_flag', 'sum'),
        )
        .reset_index()
    )
    result['usage_rate'] = result['total_suggestions_accepted'] / result['total_suggestions_sent'] * 100
    result.sort_values('usage_rate', ascending=False, inplace=True)

    # 5) Filter by usage_threshold
    result_pos = result[result['usage_rate'] >= usage_threshold].copy()

    # 6) Recompute quantiles on the filtered set
    q_low, q_high = result_pos['usage_rate'].quantile([quantiles[0], quantiles[1]]).values
    result_pos['quantile'] = result_pos['usage_rate'].apply(
        lambda x: f'Above {quantiles[1]}' if x > q_high else (
                  f'Below {quantiles[0]}' if x < q_low else 'Between')
    )

    # 7) Print counts for the new quantile buckets
    print(f"The number of territories above {quantiles[1]*100}th quantile: ",
          result_pos[result_pos['quantile'] == f'Above {quantiles[1]}'].shape[0])
    print(f"The number of territories below {quantiles[0]*100}th quantile: ",
          result_pos[result_pos['quantile'] == f'Below {quantiles[0]}'].shape[0])

    # 8) Build the box plot
    fig_usage = go.Figure()
    fig_usage.add_trace(go.Box(
        y=result['usage_rate'],
        name='Usage Rate',
        marker_color='lightblue',
        line=dict(width=3, color='red')
    ))
    fig_usage.update_layout(
        title_text='Usage Distribution',
        width=1600, height=450,
        plot_bgcolor='white', paper_bgcolor='white',
        showlegend=False,
        title_font=dict(size=18),
        margin=dict(t=120)
    )
    fig_usage.update_layout(
        annotations=[dict(text='', x=0, y=1, xref='paper', yref='paper', showarrow=False)]
    )
    if 'streamlit' not in sys.modules:
        fig_usage.show()

    # 9) Determine top & bottom groups for month-wise analysis
    top_group = result_pos.loc[result_pos['quantile'] == f'Above {quantiles[1]}', usage_level].tolist()
    bottom_group = result_pos.loc[result_pos['quantile'] == f'Below {quantiles[0]}', usage_level].tolist()

    # 10) Extract month-year and aggregate month-wise
    suggestions['month_year'] = (
        pd.to_datetime(suggestions['suggestion_generated_date'])
          .dt.to_period('M')
          .dt.to_timestamp()
    )
    intermediate = (
        suggestions
        .groupby([usage_level, 'month_year'])
        .agg(
            total_suggestions_sent       = ('suggestion_pk', 'nunique'),
            total_suggestions_sent_count = ('suggestion_pk', 'count'),
            total_suggestions_accepted   = ('accepted_flag', 'sum')
        )
        .reset_index()
    )

    top_group_month_wise    = intermediate[intermediate[usage_level].isin(top_group)]
    bottom_group_month_wise = intermediate[intermediate[usage_level].isin(bottom_group)]

    return result_pos, top_group_month_wise, bottom_group_month_wise, fig_usage


def generate_top_bottom(r, usage_level):
    top_group = r[r['quantile'].str.contains('Above')][usage_level].unique()
    bottom_group = r[r['quantile'].str.contains('Below')][usage_level].unique()

    return top_group, bottom_group


def generate_sales(r, usage_level, sales, business_unit, product_name, rec_date, sales_level = 'mini_brick', window = 3):
    # pre requisited
    top_group, bottom_group  = generate_top_bottom(r, usage_level)
    # hcps_mapping = hcps[['cust_guid', 'mini_brck_cd']]

    # deciding wt level the sales analysis needs to be done
    sales_brand  = sales[(sales.business_unit==business_unit) &(sales['product_name']==product_name)]

    print(f"→ sales_brand shape for BU={business_unit!r}, product={product_name!r}: ", sales_brand.shape)
    print("unique BUs in raw sales:", sales.business_unit.unique())
    print("unique products in raw sales:", sales.product_name.unique())



    # sales_grouped = sales.groupby([sales_level, sales['sales_month']]).agg(average_sales = ('sales_dot', np.mean)).reset_index()
    sales_grouped = sales_brand.groupby([sales_level, 'sales_month']).agg(average_sales = ('sales_dot', np.mean)).reset_index()
    # sales_grouped.head()

    # Boolean for recommendation
    sales_grouped['period'] = sales_grouped.sales_month.apply(lambda x : True if x >= pd.to_datetime(rec_date) else False)
    sales_grouped['usage'] = sales_grouped[sales_level].apply(lambda x : 'High' if x in(top_group) else ('Low' if x in(bottom_group) else 'NA'))

    display(sales_grouped.head())
    # Average sales across levels

    # filtering out inbetween groups
    sales_grouped_comparison = sales_grouped[sales_grouped['usage'].isin(['High', 'Low'])]
    # calculating the average sales across bricks for each usage group and salges month separately
    sales_grouped_average = sales_grouped_comparison.groupby(['usage','sales_month']).agg(average_sales_across_levels = ('average_sales', np.mean),
                                                                    period  = ('period', lambda x  : np.unique(x)[0])).reset_index()

    #splitting the dataframe into high and low groups
    sales_grouped_average_low = sales_grouped_average[sales_grouped_average.usage == 'Low'].reset_index(drop = True)
    sales_grouped_average_high = sales_grouped_average[sales_grouped_average.usage == 'High'].reset_index(drop = True)



    # creating a rolling mean separately for high and low groups
    sales_grouped_average_low['rolling_mean'] = sales_grouped_average_low['average_sales_across_levels'].rolling(window = window,center = False).mean()
    sales_grouped_average_high['rolling_mean'] = sales_grouped_average_high['average_sales_across_levels'].rolling(window = window,center = False).mean()


    # Dropping NA values so that they dont appear in the graph
    sales_grouped_average_low_graph = sales_grouped_average_low.dropna()
    sales_grouped_average_high_graph = sales_grouped_average_high.dropna()

    display(sales_grouped_average_low.head())
    display(sales_grouped_average_high.head())


    # # Calculating the number of hcps
    # if sales_level == 'mini_brick':
    #     col_filter = 'mini_brck_cd'
    # elif sales_level == 'brick':
    #     pass # this can be added once once the birxck level information is brought in ehre [A placeholder ;)]

    # # - Normal graph-------------------------------------------------------------------------------------------------------#

    return sales_grouped_average_low, sales_grouped_average_high, sales_grouped_average_low_graph, sales_grouped_average_high_graph, sales_grouped_average,sales_grouped


import pandas as pd
import plotly.graph_objects as go

def generate_sales_graphs(
    sales_low, sales_high,
    sales_low_graph, sales_high_graph,
    window=3,
    rec_date=None
):
    # Parse rec_date into a Python datetime
    rec_dt = pd.to_datetime(rec_date).to_pydatetime() if rec_date else None

    # ——— Figure 1: Average Sales ————————————————————————————
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(
        x=sales_low['sales_month'],
        y=sales_low['average_sales_across_levels'],
        mode='lines+markers+text',
        name='Low usage',
        text=[f"{v:.2f}" for v in sales_low['average_sales_across_levels']],
        textposition='bottom center',
        line=dict(color='blue')
    ))
    fig_avg.add_trace(go.Scatter(
        x=sales_high['sales_month'],
        y=sales_high['average_sales_across_levels'],
        mode='lines+markers+text',
        name='High usage',
        text=[f"{v:.2f}" for v in sales_high['average_sales_across_levels']],
        textposition='top center',
        line=dict(color='orange')
    ))

    # Add vertical line as a shape, spanning the plotting area:
    if rec_dt:
        fig_avg.add_shape(
            type="line",
            x0=rec_dt, x1=rec_dt,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        # And a text annotation at the top:
        fig_avg.add_annotation(
            x=rec_dt,
            y=1.02,
            xref="x", yref="paper",
            showarrow=False,
            text="🚀 Launch of recommender",
            font=dict(color="red", size=12),
            align="left"
        )

    fig_avg.update_layout(
        title_text="Average Sales",
        title_x=0.5,
        width=800, height=400,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=80, b=40, l=50, r=20),
        showlegend=True,
    )
    fig_avg.update_xaxes(title="Sales Month", tickangle=45)
    fig_avg.update_yaxes(title="Average Sales", showline=True, linecolor='black', mirror=True)

    # ——— Figure 2: Rolling Mean ————————————————————————————
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=sales_low_graph['sales_month'],           # restore x-axis
        y=sales_low_graph['rolling_mean'],
        mode='lines+markers+text',
        name='Low usage (rolling)',
        text=[f"{v:.2f}" for v in sales_low_graph['rolling_mean']],
        textposition='bottom center',
        line=dict(color='blue', dash='dash')
    ))
    fig_roll.add_trace(go.Scatter(
        x=sales_high_graph['sales_month'],
        y=sales_high_graph['rolling_mean'],
        mode='lines+markers+text',
        name='High usage (rolling)',
        text=[f"{v:.2f}" for v in sales_high_graph['rolling_mean']],
        textposition='top center',
        line=dict(color='orange', dash='dash')
    ))

    # Same vertical line + annotation:
    if rec_dt:
        fig_roll.add_shape(
            type="line",
            x0=rec_dt, x1=rec_dt,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )
        fig_roll.add_annotation(
            x=rec_dt,
            y=1.02,
            xref="x", yref="paper",
            showarrow=False,
            text="🚀 Launch of recommender",
            font=dict(color="red", size=12),
            align="left"
        )

    fig_roll.update_layout(
        title_text="Moving Average",
        title_x=0.5,
        width=800, height=400,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=80, b=40, l=50, r=20),
        showlegend=True,
    )
    fig_roll.update_xaxes(title="Sales Month", tickangle=45)
    fig_roll.update_yaxes(title="Rolling Avg Sales", showline=True, linecolor='black', mirror=True)

    return fig_avg, fig_roll

# --- Helper function for Card ---
def card_container(title):
    st.markdown(
        f"""
        <div style="background-color: #f9f9f9; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; text-align: center;">
            <h3 style="color: #333; margin: 0;">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

def blue_full_width_card(title, height_px: int = 100):
    st.markdown(
        f"""
        <div style="
            background-color: lightblue;
            color: #333;
            width: 100%;
            height: {height_px}px;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
        ">
            <h3 style="margin: 0; font-weight: 500; text-align: center;">{title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_top_n_usage(n,top_group, usage_level):

    # covnert this to string [brick and mini brick]
    if type(top_group.iloc[0][usage_level]) == str:
        pass
    else:
        top_group[usage_level] = top_group[usage_level].astype(str)

    # calculating the usage rate here
    top_group['usage_rate'] =  top_group['total_suggestions_accepted']/ top_group['total_suggestions_sent']*100
    
    display(top_group.head())

    # getting the top n bricks
    top_n_bricks = top_group.groupby(usage_level)['usage_rate'].mean().nlargest(n).index
    top_group_filtered = top_group[top_group[usage_level].isin(top_n_bricks)]
    top_group_filtered['usage_rate'] = top_group_filtered['usage_rate'].map(lambda x: f"{x:.2f}")
    # top_group_filtered = top_group_filtered.sort_values(by='usage_rate', ascending=False)


    fig = px.line(
        top_group_filtered,
        x='month_year',
        y='usage_rate',
        color=usage_level,
        line_group=usage_level,
        hover_name=usage_level,
        markers=True,
        text='usage_rate',  
        title=f'Monthly Suggestions Usage Rate by {usage_level}',
        color_discrete_sequence=px.colors.qualitative.Set3  # qualitative palette
    )


    # 4) Format the x-axis ticks
    fig.update_xaxes(
        tickformat="%b %Y",   # e.g. “Apr 2024”
        tickangle=45,
        dtick="M1",
        showgrid=False
    )

    # 5) Soften overlapping lines
    fig.update_traces(opacity=0.6, selector=dict(mode='lines+markers'))

    # 6) Final layout tweaks (still white background by default)
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=40, r=20, t=60, b=60),
        legend_title_text="Brick code",
        legend=dict(itemsizing='constant', traceorder='normal')
    )

    # fig.show()
    return top_group_filtered,fig



import pandas as pd
import numpy as np
import plotly.graph_objects as go


def project_difference_in_rolling_mean(
    sales_low_graph: pd.DataFrame,
    sales_high_graph: pd.DataFrame,
    reg_threshold=0
):
    """
    Returns:
      diff      – full difference series (DataFrame)
      history   – points up to and including rec_date (DataFrame)
      forecast  – projected series for regression_time months (DataFrame)
      viz       – concat(diff + forecast) (DataFrame)
      fig       – Plotly Figure showing history + projection
    """
    # 1) Handle empty inputs
    if sales_low_graph.empty or sales_high_graph.empty:
        diff = pd.DataFrame(columns=['sales_month', 'difference', 'period'])
        history = diff.copy()
        forecast = diff.copy()
        viz = diff.copy()
        fig = go.Figure()
        fig.update_layout(
            title="No data available for projection",
            plot_bgcolor='white', paper_bgcolor='white'
        )
        return diff, history, forecast, viz, fig

    # 2) Align both series on a full monthly index
    low_sm = sales_low_graph['sales_month'].dropna()
    high_sm = sales_high_graph['sales_month'].dropna()
    start = min(low_sm.min(), high_sm.min())
    end   = max(low_sm.max(), high_sm.max())
    all_months = pd.date_range(start=start, end=end, freq='MS')

    low_i = (
        sales_low_graph
        .set_index('sales_month')
        .reindex(all_months)
        .rename_axis('sales_month')
    )
    high_i = (
        sales_high_graph
        .set_index('sales_month')
        .reindex(all_months)
        .rename_axis('sales_month')
    )
    # fill missing
    low_i['rolling_mean'] = low_i['rolling_mean'].fillna(0)
    high_i['rolling_mean'] = high_i['rolling_mean'].fillna(0)
    low_i['period'] = low_i['period'].fillna(False)
    high_i['period'] = high_i['period'].fillna(False)

    diff = pd.DataFrame({
        'sales_month': all_months,
        'rolling_mean_low':  low_i['rolling_mean'].values,
        'rolling_mean_high': high_i['rolling_mean'].values,
        'period':            high_i['period'].values
    })
    diff['difference'] = diff['rolling_mean_high'] - diff['rolling_mean_low']

    # 3) Extract rec_date & history
    rec_idx = diff.index[diff['period']].min()
    if pd.isna(rec_idx):
        # history  = diff.copy()
        rec_date = history['sales_month'].iloc[-1]
    else:
        rec_date = diff.loc[rec_idx, 'sales_month']
        # history  = diff[diff['sales_month'] < rec_date].reset_index(drop=True)

    # 4) Determine months to project
    regression_time = int(diff['period'].sum())
    # 5) Build a “pre‐rec” slice (all months strictly before rec_date)
    pre_rec = diff[diff['sales_month'] < rec_date].reset_index(drop=True)

     # 6) Now take the last N rows of that pre_rec, where N==regression_time
    
    if regression_time > 0 and len(pre_rec) >= regression_time:
        history = pre_rec.tail(regression_time-reg_threshold).reset_index(drop=True)
    else:
        # if there aren’t enough points, just use whatever we have
        history = pre_rec.copy()

    if regression_time <= 0:
        viz = diff[['sales_month','difference']].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=viz['sales_month'],
            y=viz['difference'],
            mode='lines+markers',
            name='Δ'
        ))
        fig.update_layout(
            title="No projection (no True period)",
            plot_bgcolor='white', paper_bgcolor='white'
        )
        return diff, history, pd.DataFrame(columns=['sales_month','difference']), viz, fig

    # 5) Fit linear trend on history
    x_hist = np.arange(len(history))
    y_hist = history['difference'].to_numpy()
    m, b = np.polyfit(x_hist, y_hist, 1)

    # 6) Forecast beginning at rec_date
    rec_date = pd.to_datetime(rec_date)
    future_months = pd.date_range(
        start=rec_date,
        periods=regression_time,
        freq='MS'
    )
    future_x = np.arange(len(history), len(history) + regression_time)
    future_y = m * future_x + b
    # ensure lengths match
    if len(future_months) != len(future_y):
        future_months = pd.date_range(
            start=rec_date,
            periods=len(future_y),
            freq='MS'
        )
    forecast = pd.DataFrame({
        'sales_month': future_months,
        'difference':   future_y
    })

    # 7) Combine for viz
    viz = pd.concat([
        diff[['sales_month','difference']],
        forecast[['sales_month','difference']]
    ], ignore_index=True)

    # 8) Compute % increase
    last_real = history['difference'].iloc[-1]
    last_proj = forecast['difference'].iloc[-1]
    pct_increase = (last_real-last_proj ) / last_proj * 100

    # 9) Build Plotly figure with data labels
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=diff['sales_month'],
        y=diff['difference'],
        mode='lines+markers+text',
        name='Full Δ',
        line=dict(color='blue'),
        marker=dict(size=6),
        text=diff['difference'].round(2),
        textposition='top center'
    ))
    fig.add_trace(go.Scatter(
        x=forecast['sales_month'],
        y=forecast['difference'],
        mode='lines+markers+text',
        name='Projection Δ',
        line=dict(color='orange', dash='dot'),
        marker=dict(size=6),
        text=forecast['difference'].round(2),
        textposition='top center'
    ))

    y_min, y_max = diff['difference'].min(), diff['difference'].max()
    fig.add_shape(type="line",
        x0=rec_date, x1=rec_date,
        y0=y_min,    y1=y_max,
        xref="x",    yref="y",
        line=dict(color="red", width=2, dash="dash")
    )
    mid_idx = len(history)//2
    fig.add_annotation(
        x=history['sales_month'].iloc[mid_idx],
        y=(y_min+y_max)/2,
        text=f"Slope: {m:.2f} Δ/month",
        showarrow=False,
        font=dict(size=12), bgcolor="white", bordercolor="black"
    )
    ann_color = "darkgreen" if pct_increase >= 0 else "red"

    fig.add_annotation(
        x=forecast['sales_month'].iloc[-1],
        y=last_proj,
        text=f"{pct_increase:.1f}% since rec_date",
        showarrow=True, arrowhead=2, ax=0, ay=-30,
        font=dict(color="darkgreen", size=12),arrowcolor=ann_color,
        bgcolor="white", bordercolor="darkgreen"
    )
    fig.update_layout(
        title="Δ Rolling Mean: Full History + Projection",
        xaxis_title="Month", yaxis_title="Difference in Rolling Mean",
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(t=80,b=40,l=50,r=20)
    )
    fig.update_xaxes(tickangle=45)

    return diff, history, forecast, viz, fig
