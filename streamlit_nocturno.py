import streamlit_nocturno as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# App settings
# --------------------------------------------------
st.set_page_config(page_title="Migration Spain", layout="wide")
st.title("Migration Patterns in Spain")
st.caption(
    "An interactive app to explore the internal migration between Spanish regions. "
    "Positive net migration means a region receives more people than it loses."
)

# --------------------------------------------------
# Simple dictionaries to make charts easier to read
# --------------------------------------------------
VARIABLE_NAMES = {
    "salary": "Median salary",
    "unemployment": "Unemployment rate",
    "house_price": "House price index",
    "crime": "Crime offences",
    "net_migration": "Net migration",
    "year": "Year",
    "is_male": "Sex: male",
}

SEX_LABELS = {
    "All": "All people",
    "female": "Women",
    "male": "Men",
}

# Region coordinates. These are approximate central points for a clear visual map.
coords = {
    "Andalusia": [37.5, -4.5],
    "Aragon": [41.5, -0.9],
    "Asturias": [43.3, -5.8],
    "Balearic Islands": [39.7, 2.9],
    "Basque Country": [43.0, -2.5],
    "Canary Islands": [28.3, -15.4],
    "Cantabria": [43.2, -3.9],
    "Castile and Leon": [41.6, -4.0],
    "Castile-La Mancha": [39.5, -3.0],
    "Catalonia": [41.8, 1.5],
    "Extremadura": [39.2, -6.2],
    "Galicia": [42.7, -7.9],
    "La Rioja": [42.3, -2.4],
    "Madrid": [40.4, -3.7],
    "Murcia": [37.9, -1.5],
    "Navarre": [42.7, -1.6],
    "Valencian Community": [39.5, -0.5],
}

# --------------------------------------------------
# Load and prepare data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("migration_data.xlsx")


def add_lat_lon(data, region_col="region"):
    """Add latitude and longitude using the coords dictionary."""
    data = data.copy()
    data["lat"] = data[region_col].map(lambda x: coords[x][0])
    data["lon"] = data[region_col].map(lambda x: coords[x][1])
    return data


def readable_number(value):
    """Format big numbers for labels and metrics."""
    return f"{value:,.0f}"


df = load_data()

# Net migration by year, region and sex.
inflows = df.groupby(["year", "destination_region_name", "sex"], as_index=False)["migration_flow"].sum()
inflows.rename(columns={"destination_region_name": "region", "migration_flow": "inflow"}, inplace=True)

outflows = df.groupby(["year", "origin_region_name", "sex"], as_index=False)["migration_flow"].sum()
outflows.rename(columns={"origin_region_name": "region", "migration_flow": "outflow"}, inplace=True)

net_by_sex = pd.merge(inflows, outflows, on=["year", "region", "sex"])
net_by_sex["net_migration"] = net_by_sex["inflow"] - net_by_sex["outflow"]

# Net migration for all people combined.
total_net = net_by_sex.groupby(["year", "region"], as_index=False)["net_migration"].sum()
total_net["sex"] = "All"

# One table that works for All, female and male filters.
net_all_options = pd.concat(
    [total_net[["year", "region", "sex", "net_migration"]], net_by_sex[["year", "region", "sex", "net_migration"]]],
    ignore_index=True,
)

# Economic profile: one row per region and year.
features_list = []
for (year, region), group in df.groupby(["year", "origin_region_name"]):
    features_list.append({
        "year": year,
        "region": region,
        "salary": group["median_salary_origin"].iloc[0],
        "unemployment": group["unemployment_rate_origin"].iloc[0],
        "house_price": group["house_price_index_origin"].iloc[0],
        "crime": group["crime_offenses_origin"].iloc[0],
    })

feat_df = pd.DataFrame(features_list)
feat_df = pd.merge(feat_df, total_net[["year", "region", "net_migration"]], on=["year", "region"])

# Same regional profile, but with net migration split by sex.
sex_feat_df = pd.merge(
    feat_df.drop(columns="net_migration"),
    net_by_sex[["year", "region", "sex", "net_migration"]],
    on=["year", "region"],
)
sex_feat_df["is_male"] = np.where(sex_feat_df["sex"] == "male", 1, 0)

# --------------------------------------------------
# Sidebar menu
# --------------------------------------------------
st.sidebar.header("Menu")
page = st.sidebar.radio(
    "Go to",
    ["EDA", "Clustering", "Machine Learning", "Migration Flows"],
)

# --------------------------------------------------
# PAGE 1: EDA
# --------------------------------------------------
if page == "EDA":
    st.header("Exploratory Data Analysis")
    st.info(
        "This page gives a first look at the data. You can see which regions gain or lose people, "
        "how that changes over time, and whether patterns differ by sex."
    )

    year_map = st.slider(
        "Select year",
        int(df["year"].min()),
        int(df["year"].max()),
        int(df["year"].max()),
    )

    map_data = total_net[total_net["year"] == year_map].copy()
    map_data = add_lat_lon(map_data)
    map_data["bubble_size"] = map_data["net_migration"].abs() + 200

    st.subheader("Net migration map")
    st.write(
        "Each bubble is a region. Green means more arrivals than departures. Red means more departures than arrivals. "
        "Bubble size shows the size of the gain or loss."
    )

    fig_map = px.scatter_geo(
        map_data,
        lat="lat",
        lon="lon",
        color="net_migration",
        size="bubble_size",
        hover_name="region",
        hover_data={"net_migration": ":,.0f", "lat": False, "lon": False, "bubble_size": False},
        scope="europe",
        color_continuous_scale=["#b2182b", "#f7f7f7", "#1a9850"],
        color_continuous_midpoint=0,
        title=f"Net migration in {year_map}",
    )
    fig_map.update_geos(fitbounds="locations", visible=True)
    fig_map.update_layout(height=520, margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Migration over time")
        st.write("Use this to see if a region is gaining or losing people.")
        regions_selected = st.multiselect(
            "Regions",
            sorted(df["origin_region_name"].unique()),
            default=["Madrid", "Andalusia"],
        )
        line_data = total_net[total_net["region"].isin(regions_selected)]
        fig_line = px.line(
            line_data,
            x="year",
            y="net_migration",
            color="region",
            markers=True,
            labels={"net_migration": "Net migration", "year": "Year"},
        )
        fig_line.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_line.update_layout(height=410)
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.subheader("Sex comparison")
        st.write(
            "This compares net migration for women and men. The larger the gap, the more differences in migration between each sex."
        )
        sex_year = net_by_sex[net_by_sex["year"] == year_map].copy()
        sex_pivot = sex_year.pivot(index="region", columns="sex", values="net_migration").reset_index()
        sex_pivot["sex_gap"] = sex_pivot["female"] - sex_pivot["male"]
        sex_pivot = sex_pivot.sort_values("sex_gap")

        fig_sex = go.Figure()
        fig_sex.add_trace(go.Bar(x=sex_pivot["region"], y=sex_pivot["female"], name="Women"))
        fig_sex.add_trace(go.Bar(x=sex_pivot["region"], y=sex_pivot["male"], name="Men"))
        fig_sex.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sex.update_layout(
            barmode="group",
            height=410,
            xaxis_tickangle=-35,
            yaxis_title="Net migration",
            xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_sex, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = feat_df[["net_migration", "salary", "unemployment", "house_price", "crime"]].corr().round(2)
    corr.rename(index=VARIABLE_NAMES, columns=VARIABLE_NAMES, inplace=True)
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_corr.update_layout(height=460)
    st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------------------------------
# PAGE 2: CLUSTERING
# --------------------------------------------------
elif page == "Clustering":
    st.header("Regional Profiles - Clustering")
    st.info(
        "Clustering groups regions with similar profiles."
    )

    k = st.slider("Number of clusters", 2, 5, 3)
    features = ["salary", "unemployment", "house_price", "crime", "net_migration"]


    data_c = feat_df.groupby("region", as_index=False)[features].mean()
    scope_text = "using the average profile across all years"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_c[features])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    data_c["Cluster"] = kmeans.fit_predict(X_scaled).astype(str)

    scaled_df = pd.DataFrame(X_scaled, columns=features)
    scaled_df["region"] = data_c["region"].values
    scaled_df["Cluster"] = data_c["Cluster"].values

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    data_c["PCA 1"] = components[:, 0]
    data_c["PCA 2"] = components[:, 1]

    st.subheader("Similarity map")
    st.write(
        "Each point is a region. Regions near each other have similar profiles. "
        f"The chart is built {scope_text}."
    )

    fig_pca = px.scatter(
        data_c,
        x="PCA 1",
        y="PCA 2",
        color="Cluster",
        text="region",
        hover_name="region",
        hover_data={
            "salary": ":,.0f",
            "unemployment": ":.1f",
            "house_price": ":.1f",
            "crime": ":,.0f",
            "net_migration": ":,.0f",
            "PCA 1": False,
            "PCA 2": False,
        },
    )
    fig_pca.update_traces(textposition="top center")
    fig_pca.update_layout(height=520)
    st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("What defines each cluster?")
    st.write(
        "Bars show whether each cluster is above or below the regional average. Zero means average. "
    )

    cluster_means = scaled_df.groupby("Cluster")[features].mean().reset_index()
    cluster_long = cluster_means.melt(id_vars="Cluster", value_vars=features, var_name="Variable", value_name="Compared with average")
    cluster_long["Variable"] = cluster_long["Variable"].replace(VARIABLE_NAMES)

    fig_cluster = px.bar(
        cluster_long,
        x="Variable",
        y="Compared with average",
        color="Cluster",
        barmode="group",
        title="Cluster profile compared with the average region",
    )
    fig_cluster.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_cluster.update_layout(height=430, xaxis_title="", yaxis_title="Standardised difference")
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Explain one region")
    selected_region = st.selectbox("Choose a region", sorted(data_c["region"].unique()))
    region_values = scaled_df[scaled_df["region"] == selected_region].iloc[0]
    selected_cluster = region_values["Cluster"]
    same_cluster_regions = sorted(data_c[data_c["Cluster"] == selected_cluster]["region"].tolist())

    st.markdown(f"**{selected_region} is in Cluster {selected_cluster}.**")
    st.write("It is grouped with: " + ", ".join(same_cluster_regions) + ".")

    reasons = []
    for feature in features:
        value = float(region_values[feature])
        if abs(value) >= 0.6:
            level = "high" if value > 0 else "low"
            reasons.append(f"{VARIABLE_NAMES[feature]} is {level}")

    if len(reasons) == 0:
        reason_text = "Most variables are close to average, so the region is grouped by its overall balance rather than one extreme value."
    else:
        reason_text = "Main reason: " + ", ".join(reasons[:3]) + "."
    st.success(reason_text)

    region_plot = pd.DataFrame({
        "Variable": [VARIABLE_NAMES[f] for f in features],
        "Compared with average": [float(region_values[f]) for f in features],
    })
    region_plot["Direction"] = np.where(region_plot["Compared with average"] >= 0, "Above average", "Below average")

    fig_region = px.bar(
        region_plot,
        x="Compared with average",
        y="Variable",
        color="Direction",
        orientation="h",
        title=f"Why {selected_region} belongs to Cluster {selected_cluster}",
    )
    fig_region.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_region.update_layout(height=380, yaxis_title="", xaxis_title="Compared with the average region")
    st.plotly_chart(fig_region, use_container_width=True)

# --------------------------------------------------
# PAGE 3: MACHINE LEARNING
# --------------------------------------------------
elif page == "Machine Learning":
    st.header("Predicting Net Migration")
    st.info(
        "This page uses a simple machine learning model to estimate net migration. "
        "The aim is not only to predict, but to see which variables are most linked to migration."
    )

    ml_data = sex_feat_df.copy()

    ml_features = ["year", "salary", "unemployment", "house_price", "crime", "is_male"]

    X = ml_data[ml_features]
    y = ml_data["net_migration"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    test_model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3)
    test_model.fit(X_train, y_train)
    test_pred = test_model.predict(X_test)

    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)

    final_model = RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=3)
    final_model.fit(X, y)
    ml_data["predicted"] = final_model.predict(X)

    fit_r2 = r2_score(y, ml_data["predicted"])
    fit_rmse = np.sqrt(mean_squared_error(y, ml_data["predicted"]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical fit R²", f"{fit_r2:.3f}")
    col2.metric("Historical RMSE", f"{fit_rmse:.0f}")
    col3.metric("Test R²", f"{test_r2:.3f}")
    col4.metric("Test MAE", f"{test_mae:.0f}")

    st.caption(
        "R² close to 1 means better prediction. MAE is the average error in people. "
    )

    pred_year = st.select_slider(
        "Select year",
        options=sorted(ml_data["year"].unique()),
        value=int(ml_data["year"].max()),
    )

    sex_view = st.radio(
        "Show chart for",
        ["female", "male"],
        format_func=lambda x: SEX_LABELS[x],
        horizontal=True,
    )

    st.subheader("Actual vs predicted")
    st.write(
        "Bars show the real net migration. Diamonds show the model prediction. "
    )

    pred_year_df = ml_data[
        (ml_data["year"] == pred_year) &
        (ml_data["sex"] == sex_view)
    ].sort_values("net_migration")

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Bar(
        x=pred_year_df["region"],
        y=pred_year_df["net_migration"],
        name="Actual",
        opacity=0.75,
    ))
    fig_pred.add_trace(go.Scatter(
        x=pred_year_df["region"],
        y=pred_year_df["predicted"],
        name="Predicted",
        mode="markers",
        marker=dict(size=11, symbol="diamond"),
    ))
    fig_pred.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_pred.update_layout(
        height=430,
        xaxis_tickangle=-35,
        title=f"Actual vs predicted net migration in {pred_year} - {SEX_LABELS[sex_view]}",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="",
        yaxis_title="Net migration",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("Which variables matter most?")
    st.write(
        "This chart shows which variables the model used most."
    )

    imp = pd.DataFrame({
        "Feature": ml_features,
        "Importance": final_model.feature_importances_,
    })
    imp["Feature"] = imp["Feature"].replace(VARIABLE_NAMES)
    imp = imp.sort_values("Importance")

    fig_imp = px.bar(
        imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature importance without region",
    )
    fig_imp.update_layout(height=350, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Effect of variables")
    st.write(
        "This changes one variable while the others stay at typical values. "
    )

    col_a, col_b = st.columns(2)
    with col_a:
        pd_var = st.selectbox(
            "Select a variable to explore",
            ["salary", "unemployment", "house_price", "crime"],
            format_func=lambda x: VARIABLE_NAMES[x],
        )
    with col_b:
        selected_sex = st.radio(
            "Choose sex for the example",
            ["female", "male"],
            format_func=lambda x: SEX_LABELS[x],
            horizontal=True,
        )


    typical_profile = pd.DataFrame({
        "year": [pred_year],
        "salary": [ml_data["salary"].median()],
        "unemployment": [ml_data["unemployment"].median()],
        "house_price": [ml_data["house_price"].median()],
        "crime": [ml_data["crime"].median()],
        "is_male": [1 if selected_sex == "male" else 0],
    })

    x_range = np.linspace(ml_data[pd_var].min(), ml_data[pd_var].max(), 60)
    X_what_if = pd.concat([typical_profile] * len(x_range), ignore_index=True)
    X_what_if[pd_var] = x_range
    y_what_if = final_model.predict(X_what_if[ml_features])

    fig_what_if = px.line(
        x=x_range,
        y=y_what_if,
        labels={"x": VARIABLE_NAMES[pd_var], "y": "Predicted net migration"},
        title=f"Estimated effect of {VARIABLE_NAMES[pd_var].lower()}",
    )
    fig_what_if.update_traces(line=dict(width=3))
    fig_what_if.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_what_if.update_layout(height=350)
    st.plotly_chart(fig_what_if, use_container_width=True)

# --------------------------------------------------
# PAGE 4: MIGRATION FLOWS
# --------------------------------------------------
elif page == "Migration Flows":
    st.header("Migration Flows")
    st.info(
        "This page shows where people move from and where they go. "
        "Thicker lines mean more people moved."
    )

    year_flow = st.selectbox("Select year", sorted(df["year"].unique()), index=len(sorted(df["year"].unique())) - 1)
    top_n = 15

    flow_data = df[df["year"] == year_flow].copy()

    flow_data = flow_data[flow_data["origin_region_name"] != flow_data["destination_region_name"]]

    flows = flow_data.groupby(["origin_region_name", "destination_region_name"], as_index=False)["migration_flow"].sum()
    flows["route"] = flows["origin_region_name"] + " → " + flows["destination_region_name"]
    top_flows = flows.sort_values("migration_flow", ascending=False).head(top_n)

    biggest = top_flows.iloc[0]
    st.metric("Busiest route", biggest["route"], f"{readable_number(biggest['migration_flow'])} people")

    st.subheader("Flow map")

    max_flow = top_flows["migration_flow"].max()
    fig_flow_map = go.Figure()

    def curved_route(origin_lat, origin_lon, dest_lat, dest_lon, steps=30):
        """Create a soft curve between two map points, so overlapping routes are easier to see."""
        lats = np.linspace(origin_lat, dest_lat, steps)
        lons = np.linspace(origin_lon, dest_lon, steps)

        distance = np.sqrt((dest_lat - origin_lat) ** 2 + (dest_lon - origin_lon) ** 2)
        if distance == 0:
            return lats, lons

        curve_strength = 0.12 * distance
        curve = np.sin(np.linspace(0, np.pi, steps)) * curve_strength
        lats = lats + curve * (origin_lon - dest_lon) / distance
        lons = lons + curve * (dest_lat - origin_lat) / distance
        return lats, lons

    for _, row in top_flows.iterrows():
        origin_lat, origin_lon = coords[row["origin_region_name"]]
        dest_lat, dest_lon = coords[row["destination_region_name"]]
        width = 1 + 7 * row["migration_flow"] / max_flow
        hover_text = f"{row['origin_region_name']} → {row['destination_region_name']}<br>{row['migration_flow']:,.0f} people"

        line_lats, line_lons = curved_route(origin_lat, origin_lon, dest_lat, dest_lon)

        fig_flow_map.add_trace(go.Scattermapbox(
            lon=line_lons,
            lat=line_lats,
            mode="lines",
            line=dict(width=width, color="rgba(35, 100, 170, 0.55)"),
            hoverinfo="text",
            text=[hover_text] * len(line_lats),
            showlegend=False,
        ))

    # Small markers keep the map clean. Region names appear on hover, not as permanent labels.
    region_totals = pd.concat([
        top_flows[["origin_region_name", "migration_flow"]].rename(columns={"origin_region_name": "region"}),
        top_flows[["destination_region_name", "migration_flow"]].rename(columns={"destination_region_name": "region"}),
    ])
    region_totals = region_totals.groupby("region", as_index=False)["migration_flow"].sum()
    region_totals = add_lat_lon(region_totals)
    region_totals["hover"] = region_totals["region"]

    fig_flow_map.add_trace(go.Scattermapbox(
        lon=region_totals["lon"],
        lat=region_totals["lat"],
        mode="markers",
        marker=dict(size=10, color="rgb(20, 60, 100)", opacity=0.9),
        hoverinfo="text",
        text=region_totals["hover"],
        showlegend=False,
    ))

    fig_flow_map.update_layout(
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        title=f"Top migration routes in {year_flow}",
        mapbox=dict(
            style="carto-positron",
            center=dict(lat=region_totals["lat"].mean(), lon=region_totals["lon"].mean()),
            zoom=5,
        ),
    )
    st.plotly_chart(fig_flow_map, use_container_width=True)

    st.subheader("Main routes ranking")
    fig_flows = px.bar(
        top_flows.sort_values("migration_flow"),
        x="migration_flow",
        y="route",
        orientation="h",
        text="migration_flow",
        title=f"Top migration routes in {year_flow}",
        labels={"migration_flow": "People", "route": ""},
    )
    fig_flows.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_flows.update_layout(height=500, xaxis_title="People", yaxis_title="")
    st.plotly_chart(fig_flows, use_container_width=True)

    st.subheader("Explore one origin")
    st.write("Choose one region to see where people from that region mainly moved.")
    selected_origin = st.selectbox("Choose an origin region", sorted(flow_data["origin_region_name"].unique()))

    origin_flows = flows[flows["origin_region_name"] == selected_origin]
    origin_flows = origin_flows.sort_values("migration_flow", ascending=False).head(8)

    fig_origin = px.bar(
        origin_flows.sort_values("migration_flow"),
        x="migration_flow",
        y="destination_region_name",
        orientation="h",
        text="migration_flow",
        title=f"Main destinations from {selected_origin}",
        labels={"migration_flow": "People", "destination_region_name": "Destination"},
    )
    fig_origin.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_origin.update_layout(height=430)
    st.plotly_chart(fig_origin, use_container_width=True)

    top_destination = origin_flows.iloc[0]
    st.success(
        f"In {year_flow}, the main destination from {selected_origin} was "
        f"{top_destination['destination_region_name']} ({top_destination['migration_flow']:,.0f} people)."
    )

    st.subheader("Routes with the strongest sex difference")
    st.write(
        "This looks at whether each route is used more by women or men."
    )
    st.info(
        "The value is women minus men. Values below 0 mean more men than women moved on that route. "
        "Values above 0 mean more women than men moved on that route."
    )

    route_sex = df[
        (df["year"] == year_flow) &
        (df["origin_region_name"] != df["destination_region_name"])
    ].copy()
    route_sex = route_sex.groupby(["origin_region_name", "destination_region_name", "sex"], as_index=False)["migration_flow"].sum()
    route_sex = route_sex.pivot_table(
        index=["origin_region_name", "destination_region_name"],
        columns="sex",
        values="migration_flow",
        fill_value=0,
    ).reset_index()
    route_sex["total"] = route_sex["female"] + route_sex["male"]
    route_sex["female_share"] = route_sex["female"] / route_sex["total"]
    route_sex["sex_gap"] = route_sex["female"] - route_sex["male"]
    route_sex["route"] = route_sex["origin_region_name"] + " → " + route_sex["destination_region_name"]

    route_sex["abs_gap"] = route_sex["sex_gap"].abs()
    top_sex_gap = route_sex.sort_values("abs_gap", ascending=False).head(10).sort_values("sex_gap")

    fig_gap = px.bar(
        top_sex_gap,
        x="sex_gap",
        y="route",
        orientation="h",
        text="sex_gap",
        labels={"sex_gap": "Women minus men", "route": ""},
        title="Routes with the largest difference between women and men",
    )
    fig_gap.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_gap.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_gap.update_layout(height=480)
    st.plotly_chart(fig_gap, use_container_width=True)
