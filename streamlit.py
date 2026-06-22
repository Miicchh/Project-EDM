import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Basic settings
st.set_page_config(page_title="Migration Spain")
st.title("Migration Patterns in Spain")

# Sidebar menu
st.sidebar.header("Menu")
page = st.sidebar.radio("Go to", ["EDA", "Clustering", "Machine Learning", "Migration Flows"])
# Load data
df = pd.read_excel("migration_data.xlsx")

# Calculate net migration
inflows = df.groupby(["year", "destination_region_name", "sex"])["migration_flow"].sum().reset_index()
inflows.rename(columns={"destination_region_name": "region", "migration_flow": "inflow"}, inplace=True)
outflows = df.groupby(["year", "origin_region_name", "sex"])["migration_flow"].sum().reset_index()
outflows.rename(columns={"origin_region_name": "region", "migration_flow": "outflow"}, inplace=True)
net_df = pd.merge(inflows, outflows, on=["year", "region", "sex"])
net_df["net_migration"] = net_df["inflow"] - net_df["outflow"]
total_net = net_df.groupby(["year", "region"])["net_migration"].sum().reset_index()

# Features dataframe
features_list = []
for (y, r), group in df.groupby(["year", "origin_region_name"]):
    features_list.append({
        "year": y, 
        "region": r,
        "salary": group["median_salary_origin"].iloc[0],
        "unemployment": group["unemployment_rate_origin"].iloc[0],
        "house_price": group["house_price_index_origin"].iloc[0],
        "crime": group["crime_offenses_origin"].iloc[0]
    })
    
feat_df = pd.DataFrame(features_list)
feat_df = pd.merge(feat_df, total_net, on=["year", "region"])

# Map coordinates
coords = {
    "Andalusia": [37.5, -4.5], "Aragon": [41.5, -0.9], "Asturias": [43.3, -5.8],
    "Balearic Islands": [39.7, 2.9], "Basque Country": [43.0, -2.5],
    "Canary Islands": [28.3, -15.4], "Cantabria": [43.2, -3.9],
    "Castile and Leon": [41.6, -4.0], "Castile-La Mancha": [39.5, -3.0],
    "Catalonia": [41.8, 1.5], "Extremadura": [39.2, -6.2],
    "Galicia": [42.7, -7.9], "La Rioja": [42.3, -2.4], "Madrid": [40.4, -3.7],
    "Murcia": [37.9, -1.5], "Navarre": [42.7, -1.6], "Valencian Community": [39.5, -0.5]}

# --- PAGE 1: EDA ---
if page == "EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("Map of Net Migration")
    st.info("This map shows the net migration for each region. Green bubbles indicate a population gain, while red bubbles indicate a population loss. The size of the bubble represents the total volume of movement.")
    year_map = st.slider("Select Year", min(df["year"]), max(df["year"]), 2017)
    
    map_data = total_net[total_net["year"] == year_map].copy()
    map_data["lat"] = map_data["region"].apply(lambda x: coords[x][0])
    map_data["lon"] = map_data["region"].apply(lambda x: coords[x][1])
    map_data["size"] = abs(map_data["net_migration"]) + 100 # Avoid size 0
    
    fig_map = px.scatter_geo(
        map_data, lat="lat", lon="lon", color="net_migration", size="size",
        hover_name="region", scope="europe",
        color_continuous_scale=["red", "white", "green"],
        color_continuous_midpoint=0)
    fig_map.update_geos(fitbounds="locations")
    st.plotly_chart(fig_map)
    
    st.subheader("Migration Over Time")
    st.info("Tracks whether the selected regions are gaining or losing population year after year. Values above zero mean more arrivals than departures.")
    regions_selected = st.multiselect("Regions", df["origin_region_name"].unique(), default=["Madrid", "Andalusia"])
    line_data = total_net[total_net["region"].isin(regions_selected)]
    fig_line = px.line(line_data, x="year", y="net_migration", color="region")
    st.plotly_chart(fig_line)
    
    st.subheader("Correlation Heatmap")
    st.info("Displays how strongly variables relate to each other. Values close to 1 (blue) move in the same direction, while values close to -1 (red) move in opposite directions.")
    corr = feat_df[["net_migration", "salary", "unemployment", "house_price", "crime"]].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr)

# --- PAGE 2: CLUSTERING ---
elif page == "Clustering":
    st.header("Regional Profiles (K-Means)")
    st.info("This section groups regions into clusters based on their socioeconomic similarities (salary, unemployment, housing, crime, and migration). Regions close to each other on the plot share similar profiles.")
    year_cluster = st.selectbox("Select Year", df["year"].unique())
    k = st.slider("Number of Clusters", 2, 5, 3)
    data_c = feat_df[feat_df["year"] == year_cluster].copy()
    X = data_c[["salary", "unemployment", "house_price", "crime", "net_migration"]]
    
    # Scale and fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=k, random_state=42)
    data_c["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)
    
    # PCA to plot 2D
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    data_c["PCA 1"] = components[:, 0]
    data_c["PCA 2"] = components[:, 1]
    st.subheader("PCA Scatter Plot")
    fig_pca = px.scatter(data_c, x="PCA 1", y="PCA 2", color="Cluster", text="region")
    fig_pca.update_traces(textposition="top center")
    st.plotly_chart(fig_pca)
    st.write("Data summary by cluster:")
    st.dataframe(data_c[["region", "Cluster", "salary", "unemployment", "net_migration"]])

# --- PAGE 3: ML ---
elif page == "Machine Learning":
    st.header("Predicting Net Migration")
    st.info("A Random Forest model is trained to predict the net migration of a region based on its salary, unemployment, housing prices, and crime rate.")
    X = feat_df[["salary", "unemployment", "house_price", "crime"]]
    y = feat_df["net_migration"]
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    preds = rf.predict(X)
    
    # Basic metrics
    st.write(f"**R2 Score:** {r2_score(y, preds):.3f}")
    st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y, preds)):.2f}")
    st.subheader("Feature Importance")
    st.info("Shows which variable has the most influence on people's decision to migrate according to the model.")
    imp = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
    imp = imp.sort_values("Importance")
    fig_imp = px.bar(imp, x="Importance", y="Feature", orientation='h')
    st.plotly_chart(fig_imp)

# --- PAGE 4: FLOWS ---
elif page == " Migration Flows":
    st.header("Migration Flows")
    st.info("Highlights the busiest origin-to-destination routes. Longer bars mean more people moved between those two specific regions.")
    year_flow = st.selectbox("Select Year", df["year"].unique())
    flow_data = df[df["year"] == year_flow]
    # Remove same region
    flow_data = flow_data[flow_data["origin_region_name"] != flow_data["destination_region_name"]]
    
    # Group by route
    flows = flow_data.groupby(["origin_region_name", "destination_region_name"])["migration_flow"].sum().reset_index()
    flows["route"] = flows["origin_region_name"] + " to " + flows["destination_region_name"]
    top_flows = flows.sort_values("migration_flow", ascending=False).head(15)
    
    st.subheader("Top 15 Routes")
    fig_flows = px.bar(top_flows, x="migration_flow", y="route", orientation='h')
    st.plotly_chart(fig_flows)