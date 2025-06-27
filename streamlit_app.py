import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="📱 Smartphone Dashboard", layout="wide")
st.title("📱 Smartphone Sales Dashboard")

@st.cache_data
def load_data():
    url = "https://huggingface.co/datasets/7ng10dpE/Online-Retail/resolve/main/Smartphones_6M_FINAL.csv"
    df = pd.read_csv(url)
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    return df

df = load_data()

# Color mapping
brand_palette = px.colors.qualitative.Set3
top_brands = df['brand'].value_counts().head(20).index.tolist()
brand_colors = {b: brand_palette[i % len(brand_palette)] for i, b in enumerate(top_brands)}

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Analysis", "Brand ROI Model", "Segments & Patterns"])

# -------------------- Tab 1 --------------------
with tab1:
    st.header("🟦 Overview")

    # KPIs
    kpi_data = [
        ("Total Views", df[df['event_type'] == 'view'].shape[0], "#009688"),
        ("Total Purchases", df[df['event_type'] == 'purchase'].shape[0], "#00BCD4"),
        ("Sessions", df['user_session'].nunique() if 'user_session' in df.columns else "N/A", "#FF9800"),
        ("Users", df['user_id'].nunique(), "#F44336"),
        ("Brands", df['brand'].nunique(), "#3F51B5")
    ]
    cols = st.columns(len(kpi_data))
    for col, (label, value, color) in zip(cols, kpi_data):
        col.markdown(f"""
            <div style='background-color: {color}; padding: 20px; border-radius: 10px;
            text-align: center; color: white; height: 110px; min-height: 110px'>
                <h5 style='margin: 0;'>{label}</h5>
                <p style='font-size: 24px; line-height: 1.5; margin: 0;'><b>{int(value):,}</b></p>
            </div>
        """, unsafe_allow_html=True)

    # Top brands by purchases
    st.subheader("Top 10 Brands by Purchases")
    top10 = df[df['event_type'] == 'purchase']['brand'].value_counts().nlargest(10)
    fig1 = px.bar(top10, x=top10.index, y=top10.values, color=top10.index,
                  labels={"x": "Brand", "y": "Number of Purchases"},
                  color_discrete_map=brand_colors)
    fig1.update_traces(width=0.6)
    st.plotly_chart(fig1, use_container_width=True)

    # Conversion rate
    st.subheader("Top 10 Brands by Conversion Rate")
    views = df[df['event_type'] == 'view'].groupby('brand').size()
    purchases = df[df['event_type'] == 'purchase'].groupby('brand').size()
    conversion = (purchases / views).dropna().sort_values(ascending=False).head(10).round(3)
    fig2 = px.bar(conversion, x=conversion.index, y=conversion.values, color=conversion.index,
                  labels={"x": "Brand", "y": "Conversion Rate"},
                  color_discrete_map=brand_colors)
    fig2.update_traces(width=0.6)
    st.plotly_chart(fig2, use_container_width=True)

    # Average basket size
    st.subheader("Top 10 Brands by Average Basket Size")
    if 'basket' in df.columns:
        basket_avg = df.dropna(subset=['basket']).groupby('brand')['basket'].apply(
            lambda b: np.mean([len(eval(i)) if isinstance(i, str) else len(i) for i in b])
        ).round(1)
        basket_avg = basket_avg.sort_values(ascending=False).head(10)
        fig3 = px.bar(basket_avg, x=basket_avg.index, y=basket_avg.values, color=basket_avg.index,
                      labels={"x": "Brand", "y": "Average Basket Size"},
                      color_discrete_map=brand_colors)
        fig3.update_traces(width=0.6)
        st.plotly_chart(fig3, use_container_width=True)

    # Basket item table
    st.subheader("Top Basket Items by Frequency")
    if 'basket' in df.columns:
        all_items = df['basket'].dropna().explode()
        if isinstance(all_items.iloc[0], str):
            all_items = all_items.apply(lambda x: eval(x) if isinstance(x, str) else x)
            all_items = all_items.explode()
        item_counts = all_items.value_counts().reset_index()
        item_counts.columns = ['Product Type', 'Count']
        item_counts['Count'] = item_counts['Count'].astype(int)
        st.dataframe(
            item_counts.style.background_gradient(subset=['Count'], cmap='Oranges')
            .format({'Count': '{:,}'})
            .set_properties(subset=['Product Type'], **{'text-align': 'left'}),
            use_container_width=True
        )
    else:
        st.warning("No basket column found in dataset.")

    # Views-to-purchase ratio
    st.subheader("Views-To-Purchase Ratio Table")
    eng = df.groupby('brand')['event_type'].value_counts().unstack().fillna(0)
    eng['Views'] = eng.get('view', 0)
    eng['Add to Cart'] = eng.get('cart', 0)
    eng['Purchases'] = eng.get('purchase', 0)
    eng['Views-To-Purchase Ratio*'] = ((eng['Views'] + eng['Add to Cart']) / eng['Purchases']).replace([np.inf, -np.inf], np.nan).round(1)
    eng_table = eng[['Views', 'Add to Cart', 'Purchases', 'Views-To-Purchase Ratio*']].sort_values("Views-To-Purchase Ratio*", ascending=False)
    st.dataframe(eng_table, use_container_width=True)
    st.markdown("<p style='text-align: right;'>* (Views + Add to Cart) ÷ Purchases</p>", unsafe_allow_html=True)

    st.info("**Key Insights:**\n\n- High-basket brands like Samsung and Xiaomi stand out.\n- Some brands receive many views but low purchases — watch ROI.\n- Top co-purchased items suggest upsell bundles.")

# -------------------- Tab 2 --------------------
with tab2:
    st.header("🟧 Time Analysis")

    st.subheader("Products Viewed (Heatmap)")
    view_matrix = df[df['event_type'] == 'view'].groupby(['weekday', 'hour']).size().unstack().fillna(0)
    view_matrix = view_matrix.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig_view = go.Figure(data=go.Heatmap(
        z=view_matrix.values,
        x=view_matrix.columns,
        y=view_matrix.index,
        colorscale='Blues',
        colorbar=dict(title="Views")
    ))
    fig_view.update_layout(xaxis_title="Hour", yaxis_title="Weekday")
    st.plotly_chart(fig_view, use_container_width=True)

    st.subheader("Products Purchased (Heatmap)")
    purchase_matrix = df[df['event_type'] == 'purchase'].groupby(['weekday', 'hour']).size().unstack().fillna(0)
    purchase_matrix = purchase_matrix.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    fig_purch = go.Figure(data=go.Heatmap(
        z=purchase_matrix.values,
        x=purchase_matrix.columns,
        y=purchase_matrix.index,
        colorscale='Greens',
        colorbar=dict(title="Purchases")
    ))
    fig_purch.update_layout(xaxis_title="Hour", yaxis_title="Weekday")
    st.plotly_chart(fig_purch, use_container_width=True)

    st.subheader("Conversion Rate by Hour")
    hourly_views = df[df['event_type'] == 'view'].groupby('hour').size()
    hourly_purchases = df[df['event_type'] == 'purchase'].groupby('hour').size()
    conversion_rate = (hourly_purchases / hourly_views).dropna().round(3)
    fig_conv = px.line(x=conversion_rate.index, y=conversion_rate.values,
                       labels={'x': 'Hour of Day', 'y': 'Conversion Rate'},
                       title='Hourly Conversion Rate')
    st.plotly_chart(fig_conv, use_container_width=True)

    st.info("**Key Insights:**\n\n- Viewing spikes in the afternoon.\n- Purchases peak mid-morning (esp. Wednesday).\n- Consider scheduling ads between 9AM–12PM.")
# -------------------- Tab 3 --------------------
with tab3:
    st.header("🟩 Brand ROI Model")

    st.markdown("""
    This model estimates how likely each brand is to convert views into purchases using logistic regression.  
    Brands with high conversion and engagement are ideal candidates for ad spend.
    """)

    model_df = df[df['event_type'].isin(['view', 'cart', 'purchase'])].copy()
    model_df['converted'] = (model_df['event_type'] == 'purchase').astype(int)
    features = model_df.groupby('brand')['event_type'].value_counts().unstack().fillna(0)
    features['converted'] = model_df[model_df['converted'] == 1].groupby('brand').size()
    features = features[features['view'] > 0]
    X = features[['view', 'cart']]
    y = features['converted']

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    features['Predicted Conversion'] = model.predict_proba(X)[:, 1].round(3)
    features['Cart Rate'] = (features['cart'] / features['view']).round(3)
    features['Brand'] = features.index

    st.subheader("Predicted Conversion vs. Cart Rate")
    fig = px.scatter(
        features,
        x="Cart Rate",
        y="Predicted Conversion",
        color="Brand",
        color_discrete_map=brand_colors,
        labels={"Cart Rate": "Cart Rate", "Predicted Conversion": "Predicted Conversion Probability"}
    )
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Engagement Metrics by Brand")
    engagement = features[['view', 'cart', 'converted', 'Cart Rate', 'Predicted Conversion']]
    engagement.columns = ['Views', 'Carts', 'Purchases', 'Cart Rate', 'Predicted Conversion']
    engagement[['Views', 'Carts', 'Purchases']] = engagement[['Views', 'Carts', 'Purchases']].astype(int)
    st.dataframe(engagement.round(1), use_container_width=True)

    st.info("**Key Insights:**\n\n- Brands with high cart rate + high predicted conversion (top-right corner) are prime for promotion.\n- Several brands underperform relative to their view volume — review targeting.")

# -------------------- Tab 4 --------------------
with tab4:
    st.header("🟪 Segments & Patterns")

    st.markdown("""
    Users are grouped based on:
    - Time of activity
    - Basket diversity
    - Session engagement
    - Price sensitivity
    """)

    seg_df = df.copy()
    seg_df['hour'] = seg_df['event_time'].dt.hour
    user_features = seg_df.groupby('user_id').agg({
        'hour': 'mean',
        'price': 'mean',
        'user_session': 'nunique',
        'basket': lambda b: np.mean([len(eval(i)) if isinstance(i, str) else len(i) for i in b.dropna()])
    }).fillna(0)
    user_features.columns = ['Avg Hour', 'Avg Price', 'Sessions', 'Avg Basket Size']

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    scaled = StandardScaler().fit_transform(user_features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    user_features['Segment'] = kmeans.fit_predict(scaled)

    st.subheader("Segment Summary Table")
    summary = user_features.groupby('Segment').mean().round(1)
    st.dataframe(summary, use_container_width=True)

    st.subheader("Segment Size Distribution")
    counts = user_features['Segment'].value_counts().sort_index()
    pie_fig = px.pie(names=counts.index.astype(str), values=counts.values,
                     color=counts.index.astype(str),
                     color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(pie_fig)

    st.subheader("Average Basket Size by Segment")
    bar_fig = px.bar(
        summary,
        x=summary.index.astype(str),
        y="Avg Basket Size",
        color=summary.index.astype(str),
        color_discrete_sequence=px.colors.qualitative.Set3,
        labels={"x": "Segment", "y": "Average Basket Size"}
    )
    st.plotly_chart(bar_fig)

    st.info("**Key Insights:**\n\n- Segment 0: High-value, high-basket customers — promote bundles.\n- Segment 2: Price-sensitive, low engagement — offer discounts.\n- Segment 1: Frequent sessions — ideal for retargeting.")



