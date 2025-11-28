import streamlit as st
import pypsa
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- 1. Page Configuration ---
st.set_page_config(layout="wide", page_title="PyPSA-Earth Visualization Dashboard")

st.title("🇩🇿 PyPSA-Earth Simulation Results Visualization")
st.markdown("### Visualizing Infrastructure and Capacity Distribution")

# --- 2. Sidebar & Data Loading ---
st.sidebar.header("Configuration")

# File Paths (Defaults from test1.ipynb)
default_network_path = "results/postnetworks/elec_s_100_ec_lcopt_Co2L-1h_144h_2030_0.071_AB_0export.nc"
default_wilayas_path = "algeria_wilayas.geojson"

network_path = st.sidebar.text_input("Network File Path (.nc)", default_network_path)
wilayas_path = st.sidebar.text_input("Wilayas Shapefile (.geojson)", default_wilayas_path)

default_solar_profile = "resources/renewable_profiles/profile_solar.nc"
default_wind_profile = "resources/renewable_profiles/profile_onwind.nc"

solar_profile_path = st.sidebar.text_input("Solar Profile Path (.nc)", default_solar_profile)
wind_profile_path = st.sidebar.text_input("Wind Profile Path (.nc)", default_wind_profile)

@st.cache_resource
def load_data(net_path, wil_path):
    try:
        n = pypsa.Network(net_path)
        wilayas = gpd.read_file(wil_path)
        if wilayas.crs != 'EPSG:4326':
            wilayas = wilayas.to_crs('EPSG:4326')
        return n, wilayas
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

with st.spinner("Loading Network and Shapefiles..."):
    n, wilayas = load_data(network_path, wilayas_path)

if n is None or wilayas is None:
    st.stop()

st.sidebar.success("Data Loaded Successfully!")
st.sidebar.info(f"Buses: {len(n.buses)}")
st.sidebar.info(f"Generators: {len(n.generators)}")
st.sidebar.info(f"Links: {len(n.links)}")

# --- 3. Helper Functions ---

def get_wilaya_mapping(n, wilayas):
    # Create GeoDataFrame from buses
    buses_gdf = gpd.GeoDataFrame(
        n.buses,
        geometry=[Point(xy) for xy in zip(n.buses.x, n.buses.y)],
        crs='EPSG:4326'
    )
    
    # Spatial join
    buses_with_wilayas = gpd.sjoin(buses_gdf, wilayas, how='left', predicate='within')
    
    # Detect name column
    name_col = None
    for col in ['shapeName', 'name', 'NAME', 'NAME_1', 'NAME_2', 'wilaya', 'WILAYA']:
        if col in buses_with_wilayas.columns:
            name_col = col
            break
            
    bus_to_wilaya = {}
    if name_col:
        bus_to_wilaya = buses_with_wilayas[name_col].to_dict()
    else:
        bus_to_wilaya = {bus: bus for bus in n.buses.index}
        
    return bus_to_wilaya, name_col

bus_to_wilaya, name_col = get_wilaya_mapping(n, wilayas)

# --- 4. Visualization Logic ---

visualization_option = st.selectbox(
    "Select Visualization",
    [
        "Solar Capacity",
        "Wind Capacity",
        "Solar Potential",
        "Wind Potential",
        "Electrolyzer Capacity",
        "Electrolyzer Utilization",
        "Battery Storage",
        "Hydrogen Storage",
        "Power Balance",
        "Grid Map",
        "Combined Infrastructure",
        "CAPEX Breakdown",
        "CAPEX Breakdown",
        "Steel Cost Breakdown",
        "Solar Profile (File)",
        "Wind Profile (File)"
    ]
)

def plot_capacity_map(n, carrier, title, color_scale="Viridis", label="Capacity (MW)"):
    gens = n.generators[n.generators.carrier == carrier]
    if gens.empty:
        st.warning(f"No {carrier} generators found.")
        return

    cap_by_bus = gens.groupby('bus').p_nom_opt.sum()
    cap_by_bus = cap_by_bus[cap_by_bus > 0.1] # Filter small
    
    if cap_by_bus.empty:
        st.warning(f"No installed capacity for {carrier}.")
        return

    # Prepare data for Plotly
    data = []
    for bus, cap in cap_by_bus.items():
        data.append({
            'bus': bus,
            'lat': n.buses.at[bus, 'y'],
            'lon': n.buses.at[bus, 'x'],
            'capacity': cap,
            'wilaya': bus_to_wilaya.get(bus, "Unknown")
        })
    
    df = pd.DataFrame(data)
    
    fig = px.scatter_mapbox(
        df, 
        lat="lat", 
        lon="lon", 
        size="capacity", 
        color="capacity",
        hover_name="bus",
        hover_data={"wilaya": True, "capacity": ":.2f"},
        color_continuous_scale=color_scale,
        size_max=50, 
        zoom=4, 
        center={"lat": 36.5, "lon": 3.0},
        mapbox_style="carto-positron",
        title=title
    )
    
    # Add Wilaya boundaries
    # Note: Adding geojson directly to scatter_mapbox can be tricky with layers, 
    # but we can use update_layout to add layers if needed. 
    # For simplicity, we rely on the base map and points.
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"{title} Data:")
    st.dataframe(df.sort_values(by='capacity', ascending=False))

def plot_potential_map(n, carrier, title, color_scale="Viridis"):
    gens = n.generators[n.generators.carrier == carrier]
    if gens.empty:
        st.warning(f"No {carrier} generators found.")
        return

    if n.generators_t.p_max_pu.empty:
         st.warning(f"No time-series data (p_max_pu) found for {carrier}.")
         return

    relevant_gens = gens.index.intersection(n.generators_t.p_max_pu.columns)
    if relevant_gens.empty:
         st.warning(f"No time-series data found for existing {carrier} generators.")
         return

    # Calculate average capacity factor
    cf = n.generators_t.p_max_pu[relevant_gens].mean()
    
    # Map to buses
    cf_df = pd.DataFrame({'cf': cf, 'bus': gens.loc[relevant_gens, 'bus']})
    cf_by_bus = cf_df.groupby('bus')['cf'].mean()
    
    data = []
    for bus, val in cf_by_bus.items():
        data.append({
            'bus': bus,
            'lat': n.buses.at[bus, 'y'],
            'lon': n.buses.at[bus, 'x'],
            'capacity_factor': val,
            'wilaya': bus_to_wilaya.get(bus, "Unknown")
        })
    
    df = pd.DataFrame(data)
    
    fig = px.scatter_mapbox(
        df, 
        lat="lat", 
        lon="lon", 
        color="capacity_factor",
        hover_name="bus",
        hover_data={"wilaya": True, "capacity_factor": ":.3f"},
        color_continuous_scale=color_scale,
        zoom=4, 
        center={"lat": 36.5, "lon": 3.0},
        mapbox_style="carto-positron",
        title=title
    )
    fig.update_traces(marker=dict(size=12, opacity=0.8))
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"{title} Statistics:")
    st.write(df['capacity_factor'].describe())
    st.dataframe(df.sort_values(by='capacity_factor', ascending=False))

@st.cache_resource
def load_profile_ds(path):
    if not os.path.exists(path):
        return None
    try:
        ds = xr.open_dataset(path, chunks="auto")
        return ds
    except Exception as e:
        st.error(f"Error loading profile {path}: {e}")
        return None

def plot_profile_map(profile_path, title, color_scale="Viridis"):
    ds = load_profile_ds(profile_path)
    if ds is None:
        st.warning(f"Profile file not found: {profile_path}")
        return

    if 'profile' not in ds:
        st.warning("Variable 'profile' not found in dataset.")
        return
        
    try:
        # Calculate average capacity factor
        # Assuming dimensions are (time, bus)
        # We compute mean along the first dimension (time)
        with st.spinner(f"Processing {title}..."):
            # Check dimensions
            dims = ds['profile'].dims
            if len(dims) >= 2:
                # Usually (time, bus) or (snapshot, bus)
                time_dim = dims[0]
                cf = ds['profile'].mean(dim=time_dim).compute()
            else:
                st.warning(f"Unexpected dimensions for profile: {dims}")
                return

            # Get buses
            if 'bus' in ds:
                buses = ds['bus'].values
            else:
                # Try to use the second dimension coordinate
                buses = ds.coords[dims[1]].values
            
            # Create DataFrame
            # Ensure buses match network buses to get coordinates
            # If profile has buses not in network, we can't map them easily unless we have coords in profile
            
            # Check if profile has lat/lon
            has_coords = 'y' in ds and 'x' in ds
            
            data = []
            for i, bus in enumerate(buses):
                # Try to find coordinates
                lat, lon = None, None
                wilaya = "Unknown"
                
                if bus in n.buses.index:
                    lat = n.buses.at[bus, 'y']
                    lon = n.buses.at[bus, 'x']
                    wilaya = bus_to_wilaya.get(bus, "Unknown")
                elif has_coords:
                    # Assuming x/y are present and aligned with bus
                    # This depends on structure. Usually profiles in PyPSA-Earth are (time, bus)
                    # and don't carry x/y directly unless added.
                    pass
                
                if lat is not None and lon is not None:
                    data.append({
                        'bus': bus,
                        'lat': lat,
                        'lon': lon,
                        'capacity_factor': float(cf[i]),
                        'wilaya': wilaya
                    })
            
            if not data:
                st.warning("No profile buses matched with network buses.")
                return

            df = pd.DataFrame(data)
            
            fig = px.scatter_mapbox(
                df, 
                lat="lat", 
                lon="lon", 
                color="capacity_factor",
                hover_name="bus",
                hover_data={"wilaya": True, "capacity_factor": ":.3f"},
                color_continuous_scale=color_scale,
                zoom=4, 
                center={"lat": 36.5, "lon": 3.0},
                mapbox_style="carto-positron",
                title=title
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"{title} Statistics:")
            st.write(df['capacity_factor'].describe())
            
    except Exception as e:
        st.error(f"Error processing profile: {e}")

if visualization_option == "Solar Capacity":
    plot_capacity_map(n, 'solar', "Solar Capacity Distribution", "Solar", "Solar MW")

elif visualization_option == "Wind Capacity":
    plot_capacity_map(n, 'onwind', "Wind Capacity Distribution", "Blues", "Wind MW")

elif visualization_option == "Solar Potential":
    plot_potential_map(n, 'solar', "Solar Resource Potential (Avg. Capacity Factor)", "Inferno")

elif visualization_option == "Wind Potential":
    plot_potential_map(n, 'onwind', "Wind Resource Potential (Avg. Capacity Factor)", "Viridis")

elif visualization_option == "Electrolyzer Capacity":
    # Electrolyzers are links
    ely_links = n.links[n.links.carrier.str.contains('electrolysis', case=False)]
    if not ely_links.empty:
        cap_by_bus = ely_links.groupby('bus0').p_nom_opt.sum()
        cap_by_bus = cap_by_bus[cap_by_bus > 0.1]
        
        data = []
        for bus, cap in cap_by_bus.items():
            data.append({
                'bus': bus,
                'lat': n.buses.at[bus, 'y'],
                'lon': n.buses.at[bus, 'x'],
                'capacity': cap,
                'wilaya': bus_to_wilaya.get(bus, "Unknown")
            })
        df = pd.DataFrame(data)
        
        fig = px.scatter_mapbox(
            df, lat="lat", lon="lon", size="capacity", color="capacity",
            hover_name="bus", hover_data={"wilaya": True, "capacity": ":.2f"},
            color_continuous_scale="Purples", size_max=50, zoom=4, 
            center={"lat": 36.5, "lon": 3.0}, mapbox_style="carto-positron",
            title="Electrolyzer Capacity Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.sort_values(by='capacity', ascending=False))
    else:
        st.warning("No electrolyzers found.")

elif visualization_option == "Battery Storage":
    batteries = n.stores[n.stores.carrier == 'battery']
    if not batteries.empty:
        cap_by_bus = batteries.groupby('bus').e_nom_opt.sum()
        cap_by_bus = cap_by_bus[cap_by_bus > 0.1]
        
        data = []
        for bus, cap in cap_by_bus.items():
            # Resolve bus coordinates
            # Battery buses might be auxiliary, check if they have coords or map to AC bus
            if bus in n.buses.index:
                lat = n.buses.at[bus, 'y']
                lon = n.buses.at[bus, 'x']
            else:
                # Try to find connected bus if naming convention holds or check links
                continue
                
            data.append({
                'bus': bus,
                'lat': lat,
                'lon': lon,
                'capacity': cap,
                'wilaya': bus_to_wilaya.get(bus, "Unknown")
            })
        df = pd.DataFrame(data)
        
        fig = px.scatter_mapbox(
            df, lat="lat", lon="lon", size="capacity", color="capacity",
            hover_name="bus", hover_data={"wilaya": True, "capacity": ":.2f"},
            color_continuous_scale="Greens", size_max=50, zoom=4, 
            center={"lat": 36.5, "lon": 3.0}, mapbox_style="carto-positron",
            title="Battery Storage Distribution (MWh)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.sort_values(by='capacity', ascending=False))
    else:
        st.warning("No battery storage found.")

elif visualization_option == "Hydrogen Storage":
    h2_stores = n.stores[n.stores.carrier == 'H2']
    if not h2_stores.empty:
        cap_by_bus = h2_stores.groupby('bus').e_nom_opt.sum()
        cap_by_bus = cap_by_bus[cap_by_bus > 0.1]
        
        data = []
        for bus, cap in cap_by_bus.items():
            if bus in n.buses.index:
                lat = n.buses.at[bus, 'y']
                lon = n.buses.at[bus, 'x']
            else:
                continue
            data.append({
                'bus': bus,
                'lat': lat,
                'lon': lon,
                'capacity': cap,
                'wilaya': bus_to_wilaya.get(bus, "Unknown")
            })
        df = pd.DataFrame(data)
        
        fig = px.scatter_mapbox(
            df, lat="lat", lon="lon", size="capacity", color="capacity",
            hover_name="bus", hover_data={"wilaya": True, "capacity": ":.2f"},
            color_continuous_scale="RdPu", size_max=50, zoom=4, 
            center={"lat": 36.5, "lon": 3.0}, mapbox_style="carto-positron",
            title="Hydrogen Storage Distribution (MWh)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.sort_values(by='capacity', ascending=False))
    else:
        st.warning("No hydrogen storage found.")

elif visualization_option == "Electrolyzer Utilization":
    st.subheader("Electrolyzer Utilization Duration Curve")
    ely_links = n.links[n.links.carrier.str.contains("electrolysis", case=False)]
    if ely_links.empty:
        st.warning("No electrolyzers found.")
    else:
        # Check if time series data exists
        if n.links_t.p0.empty:
             st.warning("No time-series data found for electrolyzers.")
        else:
            # Sum power consumption across all electrolyzers
            # p0 is input (consumption)
            ely_p = n.links_t.p0[ely_links.index].sum(axis=1)
            
            total_cap = ely_links.p_nom_opt.sum()
            if total_cap > 0.1:
                utilization = ely_p / total_cap * 100
                # Sort descending for duration curve
                duration_curve = utilization.sort_values(ascending=False).reset_index(drop=True)
                
                # Create DataFrame for Plotly
                df_dur = pd.DataFrame({
                    "Hours": range(len(duration_curve)),
                    "Utilization (%)": duration_curve.values
                })
                
                fig = px.line(df_dur, x="Hours", y="Utilization (%)", title="Electrolyzer Utilization Duration Curve")
                fig.update_layout(xaxis_title="Hours", yaxis_title="Utilization (%)")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("Total Electrolyzer Capacity", f"{total_cap:.2f} MW")
                col2.metric("Average Capacity Factor", f"{utilization.mean():.2f} %")
            else:
                st.warning("Electrolyzer capacity is negligible.")

elif visualization_option == "Power Balance":
    st.subheader("System Power Balance (Representative Week)")
    
    # Select a week with high activity (e.g., max load or max generation)
    # For simplicity, we'll pick the week with the highest total generation
    if n.generators_t.p.empty:
        st.warning("No generation time-series data found.")
    else:
        total_gen = n.generators_t.p.sum(axis=1)
        if total_gen.empty:
             st.warning("No generation data available.")
        else:
            peak_time = total_gen.idxmax()
            # Get a window around the peak (e.g., 1 week)
            # Assuming hourly resolution, 168 hours
            # Handle indices properly
            try:
                start_idx = max(0, n.snapshots.get_loc(peak_time) - 84)
                end_idx = min(len(n.snapshots), start_idx + 168)
                time_window = n.snapshots[start_idx:end_idx]
                
                # Prepare data
                # Generation by carrier
                gen_data = n.generators_t.p.loc[time_window].groupby(n.generators.carrier, axis=1).sum()
                
                # Storage discharge (positive supply)
                store_dis = n.stores_t.p.loc[time_window]
                # Separate discharge (positive) and charge (negative) if p represents net flow
                # In PyPSA, stores_t.p is positive for discharge? Let's check convention.
                # Usually: p > 0 is discharge (supply), p < 0 is charge (demand)
                # But we need to group by carrier
                store_flow = store_dis.groupby(n.stores.carrier, axis=1).sum()
                store_supply = store_flow.clip(lower=0)
                store_demand = store_flow.clip(upper=0) # Negative
                
                # Demand (Load + Electrolyzers)
                # Loads
                if not n.loads_t.p_set.empty:
                    load = n.loads_t.p_set.loc[time_window].sum(axis=1)
                else:
                    load = pd.Series(0, index=time_window)
                
                # Electrolyzers (Links)
                ely_links = n.links[n.links.carrier.str.contains("electrolysis", case=False)]
                if not ely_links.empty:
                    ely_demand = n.links_t.p0.loc[time_window, ely_links.index].sum(axis=1)
                else:
                    ely_demand = pd.Series(0, index=time_window)
                    
                # Combine Supply
                supply_df = pd.concat([gen_data, store_supply], axis=1)
                # Filter small columns
                supply_df = supply_df.loc[:, (supply_df.sum() > 1)]
                
                # Create figure
                fig = go.Figure()
                
                # Add supply traces (stacked area)
                for col in supply_df.columns:
                    fig.add_trace(go.Scatter(
                        x=supply_df.index, y=supply_df[col],
                        mode='lines',
                        name=f"Supply: {col}",
                        stackgroup='one'
                    ))
                    
                # Add demand traces (lines or negative area)
                # Let's plot demand as lines for comparison
                fig.add_trace(go.Scatter(
                    x=time_window, y=load,
                    mode='lines', name='Electric Load',
                    line=dict(color='black', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=time_window, y=ely_demand,
                    mode='lines', name='Electrolyzer Demand',
                    line=dict(color='purple', width=2, dash='dash')
                ))
                
                # Add storage charging as negative area if significant
                if store_demand.sum().sum() < -1:
                     for col in store_demand.columns:
                        if store_demand[col].sum() < -1:
                            fig.add_trace(go.Scatter(
                                x=store_demand.index, y=store_demand[col],
                                mode='lines',
                                name=f"Charge: {col}",
                                stackgroup='two', # Separate stack group for negative
                                fill='tozeroy'
                            ))

                fig.update_layout(title="Power Balance (Peak Week)", yaxis_title="Power (MW)")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error plotting power balance: {e}")

elif visualization_option == "CAPEX Breakdown":
    st.subheader("Total Capital Expenditure Breakdown")
    
    costs = {}
    # Generators
    for carrier in n.generators.carrier.unique():
        c = (n.generators.p_nom_opt[n.generators.carrier == carrier] * n.generators.capital_cost[n.generators.carrier == carrier]).sum()
        if c > 1: costs[carrier] = c
    
    # Links
    for carrier in n.links.carrier.unique():
        c = (n.links.p_nom_opt[n.links.carrier == carrier] * n.links.capital_cost[n.links.carrier == carrier]).sum()
        if c > 1: costs[carrier] = c
        
    # Stores
    for carrier in n.stores.carrier.unique():
        c = (n.stores.e_nom_opt[n.stores.carrier == carrier] * n.stores.capital_cost[n.stores.carrier == carrier]).sum()
        if c > 1: costs[carrier] = c
        
    # Transmission
    if not n.lines.empty:
        c = (n.lines.s_nom_opt * n.lines.capital_cost).sum()
        if c > 1: costs['Transmission'] = c
        
    if not costs:
        st.warning("No CAPEX data found.")
    else:
        df_costs = pd.DataFrame(list(costs.items()), columns=['Component', 'CAPEX'])
        
        fig = px.pie(df_costs, values='CAPEX', names='Component', title="CAPEX Breakdown by Technology")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("CAPEX Data (EUR):")
        st.dataframe(df_costs.sort_values(by='CAPEX', ascending=False))

elif visualization_option == "Steel Cost Breakdown":
    st.subheader("Green Steel Production Cost Breakdown")
    
    # 1. Find Steel Load
    # Search for loads with "Steel" in the name
    steel_loads = n.loads.index[n.loads.index.str.contains("Steel", case=False)]
    
    if steel_loads.empty:
        st.warning("No 'Steel' load found in the network.")
    else:
        # Allow user to select if multiple steel loads exist
        load_name = st.selectbox("Select Steel Plant Load", steel_loads)
        
        # Get the bus associated with this load
        bus_name = n.loads.at[load_name, "bus"]
        
        # Check if marginal prices are available
        if n.buses_t.marginal_price.empty:
            st.error("No marginal price data available in the network. Cannot calculate costs.")
        elif bus_name not in n.buses_t.marginal_price.columns:
             st.error(f"No marginal price found for bus '{bus_name}'.")
        else:
            # 2. Get Hydrogen Price (LCOH)
            # The load is on the H2 bus, so its marginal price is the H2 price
            h2_prices = n.buses_t.marginal_price[bus_name]
            
            # 3. Get Load Profile
            if load_name in n.loads_t.p_set.columns:
                load_profile = n.loads_t.p_set[load_name]
            else:
                # Static load
                static_val = n.loads.at[load_name, "p_set"]
                load_profile = pd.Series(static_val, index=n.snapshots)
            
            # 4. Calculate Weighted Average LCOH
            total_load = load_profile.sum()
            if total_load > 0:
                lcoh = (h2_prices * load_profile).sum() / total_load
            else:
                lcoh = 0
                
            # 5. Get Electricity Price (LCOE)
            # Assume AC bus name is derived by removing " H2" suffix if present
            # This is a heuristic based on PyPSA-Earth-Sec naming conventions
            ac_bus_name = bus_name.replace(" H2", "").replace(" H2", "") 
            
            if ac_bus_name in n.buses_t.marginal_price.columns:
                elec_prices = n.buses_t.marginal_price[ac_bus_name]
                lcoe = elec_prices.mean() # Simple average for base load assumption
            else:
                st.warning(f"Could not automatically find AC bus '{ac_bus_name}' for electricity pricing. Assuming 0 for electricity cost.")
                lcoe = 0
                
            # 6. Calculate Component Costs (Assumptions from notebook)
            h2_per_ton = 1.7 # MWh H2 per ton steel
            elec_per_ton = 0.4 # MWh Elec per ton steel
            non_energy_cost = 300.0 # EUR/ton (Iron Ore, CAPEX, Labor, etc.)
            
            cost_h2 = lcoh * h2_per_ton
            cost_elec = lcoe * elec_per_ton
            
            total_cost = cost_h2 + cost_elec + non_energy_cost
            
            # 7. Plot
            cost_data = pd.DataFrame({
                "Component": ["Hydrogen", "Electricity", "Non-Energy"],
                "Cost (EUR/ton)": [cost_h2, cost_elec, non_energy_cost],
                "Description": [f"@ {lcoh:.2f} EUR/MWh", f"@ {lcoe:.2f} EUR/MWh", "Iron Ore, CAPEX, etc."]
            })
            
            fig = px.bar(
                cost_data, 
                x="Component", 
                y="Cost (EUR/ton)", 
                title=f"Green Steel Cost Structure ({load_name})",
                text="Cost (EUR/ton)",
                color="Component",
                color_discrete_sequence=["#2ca02c", "#1f77b4", "#7f7f7f"],
                hover_data=["Description"]
            )
            fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(yaxis_title="Cost (EUR/ton steel)", showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display Key Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Steel Cost", f"{total_cost:.2f} €/ton")
            c2.metric("LCOH (Hydrogen)", f"{lcoh:.2f} €/MWh")
            c3.metric("LCOE (Electricity)", f"{lcoe:.2f} €/MWh")

elif visualization_option == "Solar Profile (File)":
    plot_profile_map(solar_profile_path, "Solar Profile Potential (Avg. CF)", "Inferno")

elif visualization_option == "Wind Profile (File)":
    plot_profile_map(wind_profile_path, "Wind Profile Potential (Avg. CF)", "Viridis")

elif visualization_option == "Grid Map":
    st.subheader("Electrical Grid Map")
    
    edge_x = []
    edge_y = []
    for idx, row in n.lines.iterrows():
        x0 = n.buses.at[row.bus0, 'x']
        y0 = n.buses.at[row.bus0, 'y']
        x1 = n.buses.at[row.bus1, 'x']
        y1 = n.buses.at[row.bus1, 'y']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lat=edge_y, lon=edge_x,
        mode='lines',
        line=dict(width=1, color='gray'),
        name='Transmission Lines',
        hoverinfo='none'
    ))

    fig.add_trace(go.Scattermapbox(
        lat=n.buses.y, lon=n.buses.x,
        mode='markers',
        marker=dict(size=5, color='black'),
        text=n.buses.index,
        name='Substations'
    ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=36.5, lon=3.0), zoom=4),
        height=600,
        margin={"r":0,"t":0,"l":0,"b":0}
    )
    st.plotly_chart(fig, use_container_width=True)

elif visualization_option == "Combined Infrastructure":
    st.subheader("Combined Infrastructure Map")
    
    fig = go.Figure()
    
    # 1. Grid Lines
    edge_x = []
    edge_y = []
    for idx, row in n.lines.iterrows():
        x0 = n.buses.at[row.bus0, 'x']
        y0 = n.buses.at[row.bus0, 'y']
        x1 = n.buses.at[row.bus1, 'x']
        y1 = n.buses.at[row.bus1, 'y']
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    fig.add_trace(go.Scattermapbox(
        lat=edge_y, lon=edge_x,
        mode='lines',
        line=dict(width=1, color='gray'),
        name='Grid',
        hoverinfo='none'
    ))
    
    # 2. Solar
    solar_gens = n.generators[n.generators.carrier == 'solar']
    if not solar_gens.empty:
        s_bus = solar_gens.groupby('bus').p_nom_opt.sum()
        s_bus = s_bus[s_bus > 1]
        if not s_bus.empty:
            fig.add_trace(go.Scattermapbox(
                lat=[n.buses.at[b, 'y'] for b in s_bus.index],
                lon=[n.buses.at[b, 'x'] for b in s_bus.index],
                mode='markers',
                marker=dict(size=[v/s_bus.max()*30 + 5 for v in s_bus.values], color='gold', opacity=0.7),
                name='Solar',
                text=[f"{v:.0f} MW" for v in s_bus.values]
            ))

    # 3. Wind
    wind_gens = n.generators[n.generators.carrier == 'onwind']
    if not wind_gens.empty:
        w_bus = wind_gens.groupby('bus').p_nom_opt.sum()
        w_bus = w_bus[w_bus > 1]
        if not w_bus.empty:
            fig.add_trace(go.Scattermapbox(
                lat=[n.buses.at[b, 'y'] for b in w_bus.index],
                lon=[n.buses.at[b, 'x'] for b in w_bus.index],
                mode='markers',
                marker=dict(size=[v/w_bus.max()*30 + 5 for v in w_bus.values], color='skyblue', opacity=0.7),
                name='Wind',
                text=[f"{v:.0f} MW" for v in w_bus.values]
            ))
            
    # 4. Electrolyzers
    ely_links = n.links[n.links.carrier.str.contains('electrolysis', case=False)]
    if not ely_links.empty:
        e_bus = ely_links.groupby('bus0').p_nom_opt.sum()
        e_bus = e_bus[e_bus > 1]
        if not e_bus.empty:
            fig.add_trace(go.Scattermapbox(
                lat=[n.buses.at[b, 'y'] for b in e_bus.index],
                lon=[n.buses.at[b, 'x'] for b in e_bus.index],
                mode='markers',
                marker=dict(size=[v/e_bus.max()*30 + 5 for v in e_bus.values], color='purple', opacity=0.7),
                name='Electrolyzer',
                text=[f"{v:.0f} MW" for v in e_bus.values]
            ))

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(center=dict(lat=36.5, lon=3.0), zoom=4),
        height=700,
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("Generated by PyPSA-Earth Visualization App")
