import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(page_title="V1-V2 Compare Dash", layout="wide")

st.title("📊 V1-V2 Compare Dash")
st.markdown("---")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Define markets to exclude
        excluded_markets = [
            'Total Game 1st Downs',
            'Team to Call First Timeout',
            'Team to have First Coaches Challenge',
            'Total Field Goals',
            'Total Match Sacks',
            'Total Game Turnovers',
            '1st Half Total Field Goals',
            '1st Half Total Home Field Goals',
            '1st Half Total Away Field Goals',
            'Interceptions',
            'Alt Interceptions',
            'Tackles & Assists',
            'To Make a Sack',
            'Kicking Points',
            'Alt Kicking Points',
            'Home Team Total 1st Downs',
            'Away Team Total 1st Downs',
            'Home Total TDs',
            'Away Total TDs',
            'Home Total Field Goals',
            'Away Total Field Goals',
            'Home Total Sacks',
            'Away Total Sacks',
            'Home Team Total Turnovers',
            'Away Team Total Turnovers'
        ]
        
        # Remove excluded markets (case-insensitive partial matching)
        if 'Market' in df.columns:
            mask = df['Market'].str.contains('|'.join(excluded_markets), case=False, na=False)
            df = df[~mask]
        
        # Remove rows where both V1 Odds and V2 Odds are N/A
        df = df[(df['V1 Odds'] != 'N/A') & (df['V2 Odds'] != 'N/A')]
        
        # Convert odds columns to numeric
        df['V1 Odds'] = pd.to_numeric(df['V1 Odds'], errors='coerce')
        df['V2 Odds'] = pd.to_numeric(df['V2 Odds'], errors='coerce')
        
        # Calculate V2 - V1 difference
        df['V2_minus_V1'] = df['V2'] - df['V1']
        
        # Extract player names and betting lines
        def extract_player_and_line(text):
            if pd.isna(text) or text == 'N/A':
                return None, None
            
            text = str(text).strip()
            
            # Pattern: (numbers) PlayerName number/decimal+/-
            # Split by closing parenthesis
            parts = text.split(')')
            if len(parts) < 2:
                return None, None
            
            # Get everything after the parenthesis
            remaining = parts[1].strip()
            
            # Extract betting line (number with optional decimal and +/-)
            line_match = remaining.split()
            if len(line_match) < 2:
                return remaining if remaining else None, None
            
            # Last part should be the betting line
            betting_line = line_match[-1]
            
            # Player name is everything except the last part
            player_name = ' '.join(line_match[:-1]).strip()
            
            return player_name if player_name else None, betting_line if betting_line else None
        
        # Extract clean selection names (remove parenthesis numbers)
        def extract_clean_selection(text):
            if pd.isna(text) or text == 'N/A':
                return None
            
            text = str(text).strip()
            
            # Pattern: (numbers) Name with betting line
            # Split by closing parenthesis and take everything after
            parts = text.split(')')
            if len(parts) < 2:
                return text  # Return original if no parenthesis found
            
            # Get everything after the parenthesis and clean it up
            clean_name = parts[1].strip()
            
            return clean_name if clean_name else text
        
        # Parse timestamp to get quarter and time
        def parse_timestamp(timestamp):
            if pd.isna(timestamp) or timestamp == 'N/A':
                return None, None, None
            
            timestamp = str(timestamp).strip()
            
            # Extract quarter (Q0, Q1, Q2, Q3, Q4, etc.)
            quarter_match = timestamp.split(' - ')
            if len(quarter_match) != 2:
                return None, None, None
            
            quarter = quarter_match[0].strip()
            time_str = quarter_match[1].strip()
            
            # Convert time to minutes for sorting (15:00 = 15.0, 10:30 = 10.5)
            try:
                time_parts = time_str.split(':')
                if len(time_parts) == 2:
                    minutes = int(time_parts[0])
                    seconds = int(time_parts[1])
                    time_decimal = minutes + (seconds / 60.0)
                    
                    # Create a combined time value for plotting
                    # Q0=0, Q1=15, Q2=30, Q3=45, Q4=60, etc.
                    quarter_num = 0
                    if quarter.startswith('Q') and len(quarter) > 1:
                        try:
                            quarter_num = int(quarter[1:])
                        except:
                            quarter_num = 0
                    
                    # Combined time: quarter_num * 15 + (15 - time_decimal)
                    # This makes each quarter span 15 minutes
                    combined_time = quarter_num * 15 + (15 - time_decimal)
                    
                    return quarter, time_decimal, combined_time
                else:
                    return quarter, None, None
            except:
                return quarter, None, None
        
        # Apply timestamp parsing
        if 'Timestamp' in df.columns:
            df[['Quarter', 'Time_Minutes', 'Combined_Time']] = df['Timestamp'].apply(
                lambda x: pd.Series(parse_timestamp(x))
            )
        
        # Apply extraction to V1 and V2 Selection columns
        if 'V1 Selection' in df.columns:
            df[['V1_Player', 'V1_Line']] = df['V1 Selection'].apply(
                lambda x: pd.Series(extract_player_and_line(x))
            )
            df['V1_Clean_Selection'] = df['V1 Selection'].apply(extract_clean_selection)
        
        if 'V2 Selection' in df.columns:
            df[['V2_Player', 'V2_Line']] = df['V2 Selection'].apply(
                lambda x: pd.Series(extract_player_and_line(x))
            )
            df['V2_Clean_Selection'] = df['V2 Selection'].apply(extract_clean_selection)
        
        # Replace 'N/A' with actual None/NaN for filtering
        filter_columns = ['Market', 'V2 Selection', 'Timestamp', 'Down Number', 
                         'Yards to go', 'Yards to end zone', 'Team in possession', 
                         'Home Score', 'Away Score']
        
        for col in filter_columns:
            if col in df.columns:
                df[col] = df[col].replace('N/A', pd.NA)
        
        return df
    
    df = load_data(uploaded_file)
    
    st.success(f"Data loaded successfully! {len(df)} rows after removing N/A odds values.")
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filters")
    
    # Initialize filtered dataframe
    filtered_df = df.copy()
    
    # Market filter with search
    if 'Market' in df.columns:
        st.sidebar.subheader("Market")
        market_search = st.sidebar.text_input("Search Markets:", key="market_search")
        
        available_markets = df['Market'].dropna().unique()
        if market_search:
            available_markets = [m for m in available_markets if market_search.lower() in str(m).lower()]
        
        selected_markets = st.sidebar.multiselect(
            "Select Markets:",
            options=available_markets,
            default=available_markets,
            key="market_filter"
        )
        
        if selected_markets:
            filtered_df = filtered_df[filtered_df['Market'].isin(selected_markets)]
    
    # Market Number Exclusion Filter
    if 'Market' in df.columns:
        st.sidebar.subheader("Exclude Market Numbers")
        
        # Extract unique market numbers from market names
        def extract_market_number(market_name):
            if pd.isna(market_name):
                return None
            market_str = str(market_name).strip()
            # Look for pattern like "(1050)" at the beginning
            if market_str.startswith('(') and ')' in market_str:
                try:
                    number_part = market_str.split(')')[0][1:]  # Remove ( and )
                    return number_part
                except:
                    return None
            return None
        
        # Get all unique market numbers
        market_numbers = []
        for market in filtered_df['Market'].dropna().unique():
            number = extract_market_number(market)
            if number:
                market_numbers.append(number)
        
        unique_market_numbers = sorted(list(set(market_numbers)))
        
        if unique_market_numbers:
            excluded_numbers = st.sidebar.multiselect(
                "Select Market Numbers to EXCLUDE:",
                options=unique_market_numbers,
                default=[],
                key="exclude_market_numbers",
                help="Select market numbers to remove from analysis (e.g., select '1050' to remove all markets starting with '(1050)')"
            )
            
            # Apply exclusion filter
            if excluded_numbers:
                for exclude_num in excluded_numbers:
                    filtered_df = filtered_df[~filtered_df['Market'].str.contains(f'\\({exclude_num}\\)', na=False, regex=True)]
                
                st.sidebar.write(f"Excluded {len(excluded_numbers)} market number(s)")
        else:
            st.sidebar.write("No market numbers found to exclude")
    
    # V2 Selection filter with search (using clean selection names)
    if 'V2_Clean_Selection' in df.columns:
        st.sidebar.subheader("V2 Selection")
        v2_search = st.sidebar.text_input("Search V2 Selections:", key="v2_search")
        
        available_v2 = filtered_df['V2_Clean_Selection'].dropna().unique()
        if v2_search:
            available_v2 = [v for v in available_v2 if v2_search.lower() in str(v).lower()]
        
        selected_v2 = st.sidebar.multiselect(
            "Select V2 Selections:",
            options=available_v2,
            default=available_v2,
            key="v2_filter"
        )
        
        if selected_v2:
            filtered_df = filtered_df[filtered_df['V2_Clean_Selection'].isin(selected_v2)]
    
    # Player filter with search (V2_Player)
    if 'V2_Player' in df.columns:
        st.sidebar.subheader("Player")
        player_search = st.sidebar.text_input("Search Players:", key="player_search")
        
        available_players = filtered_df['V2_Player'].dropna().unique()
        if player_search:
            available_players = [p for p in available_players if player_search.lower() in str(p).lower()]
        
        selected_players = st.sidebar.multiselect(
            "Select Players:",
            options=available_players,
            default=available_players,
            key="player_filter"
        )
        
        if selected_players:
            filtered_df = filtered_df[filtered_df['V2_Player'].isin(selected_players)]
    
    # Betting Line filter (V2_Line)
    if 'V2_Line' in df.columns:
        st.sidebar.subheader("Betting Line")
        available_lines = filtered_df['V2_Line'].dropna().unique()
        selected_lines = st.sidebar.multiselect(
            "Select Betting Lines:",
            options=sorted(available_lines),
            default=available_lines,
            key="line_filter"
        )
        
        if selected_lines:
            filtered_df = filtered_df[filtered_df['V2_Line'].isin(selected_lines)]
    
    # Quarter filter (from parsed timestamps)
    if 'Quarter' in df.columns:
        st.sidebar.subheader("Quarter")
        available_quarters = filtered_df['Quarter'].dropna().unique()
        # Sort quarters properly (Q0, Q1, Q2, Q3, Q4, etc.)
        available_quarters = sorted(available_quarters, key=lambda x: (len(x), x))
        selected_quarters = st.sidebar.multiselect(
            "Select Quarters:",
            options=available_quarters,
            default=available_quarters,
            key="quarter_filter"
        )
        
        if selected_quarters:
            filtered_df = filtered_df[filtered_df['Quarter'].isin(selected_quarters)]
    
    # Timestamp filter
    if 'Timestamp' in df.columns:
        st.sidebar.subheader("Timestamp")
        available_timestamps = filtered_df['Timestamp'].dropna().unique()
        selected_timestamps = st.sidebar.multiselect(
            "Select Timestamps:",
            options=available_timestamps,
            default=available_timestamps,
            key="timestamp_filter"
        )
        
        if selected_timestamps:
            filtered_df = filtered_df[filtered_df['Timestamp'].isin(selected_timestamps)]
    
    # Down Number filter
    if 'Down Number' in df.columns:
        st.sidebar.subheader("Down Number")
        available_downs = filtered_df['Down Number'].dropna().unique()
        selected_downs = st.sidebar.multiselect(
            "Select Down Numbers:",
            options=available_downs,
            default=available_downs,
            key="down_filter"
        )
        
        if selected_downs:
            filtered_df = filtered_df[filtered_df['Down Number'].isin(selected_downs)]
    
    # Yards to go filter
    if 'Yards to go' in df.columns:
        st.sidebar.subheader("Yards to Go")
        # Check if we have numeric data for yards to go
        yards_to_go_numeric = pd.to_numeric(filtered_df['Yards to go'], errors='coerce').dropna()
        if len(yards_to_go_numeric) > 0:
            min_yards = int(yards_to_go_numeric.min())
            max_yards = int(yards_to_go_numeric.max())
            if min_yards != max_yards:  # Only show slider if there's a range
                selected_yards_range = st.sidebar.slider(
                    "Yards to Go Range:",
                    min_value=min_yards,
                    max_value=max_yards,
                    value=(min_yards, max_yards),
                    key="yards_to_go_filter"
                )
                # Filter based on numeric range
                numeric_yards = pd.to_numeric(filtered_df['Yards to go'], errors='coerce')
                filtered_df = filtered_df[
                    (numeric_yards >= selected_yards_range[0]) & 
                    (numeric_yards <= selected_yards_range[1]) |
                    filtered_df['Yards to go'].isna()  # Keep N/A values
                ]
            else:
                st.sidebar.write(f"All values: {min_yards} yards")
        else:
            # Fallback to multiselect if no numeric data
            available_yards = filtered_df['Yards to go'].dropna().unique()
            selected_yards = st.sidebar.multiselect(
                "Select Yards to Go:",
                options=available_yards,
                default=available_yards,
                key="yards_filter"
            )
            
            if selected_yards:
                filtered_df = filtered_df[filtered_df['Yards to go'].isin(selected_yards)]
    
    # Yards to end zone filter
    if 'Yards to end zone' in df.columns:
        st.sidebar.subheader("Yards to End Zone")
        # Check if we have numeric data for yards to end zone
        yards_endzone_numeric = pd.to_numeric(filtered_df['Yards to end zone'], errors='coerce').dropna()
        if len(yards_endzone_numeric) > 0:
            min_endzone = int(yards_endzone_numeric.min())
            max_endzone = int(yards_endzone_numeric.max())
            if min_endzone != max_endzone:  # Only show slider if there's a range
                selected_endzone_range = st.sidebar.slider(
                    "Yards to End Zone Range:",
                    min_value=min_endzone,
                    max_value=max_endzone,
                    value=(min_endzone, max_endzone),
                    key="yards_endzone_filter"
                )
                # Filter based on numeric range
                numeric_endzone = pd.to_numeric(filtered_df['Yards to end zone'], errors='coerce')
                filtered_df = filtered_df[
                    (numeric_endzone >= selected_endzone_range[0]) & 
                    (numeric_endzone <= selected_endzone_range[1]) |
                    filtered_df['Yards to end zone'].isna()  # Keep N/A values
                ]
            else:
                st.sidebar.write(f"All values: {min_endzone} yards")
        else:
            # Fallback to multiselect if no numeric data
            available_endzone = filtered_df['Yards to end zone'].dropna().unique()
            selected_endzone = st.sidebar.multiselect(
                "Select Yards to End Zone:",
                options=available_endzone,
                default=available_endzone,
                key="endzone_filter"
            )
            
            if selected_endzone:
                filtered_df = filtered_df[filtered_df['Yards to end zone'].isin(selected_endzone)]
    
    # Team in possession filter
    if 'Team in possession' in df.columns:
        st.sidebar.subheader("Team in Possession")
        available_teams = filtered_df['Team in possession'].dropna().unique()
        selected_teams = st.sidebar.multiselect(
            "Select Teams in Possession:",
            options=available_teams,
            default=available_teams,
            key="team_filter"
        )
        
        if selected_teams:
            filtered_df = filtered_df[filtered_df['Team in possession'].isin(selected_teams)]
    
    # Home Score filter
    if 'Home Score' in df.columns:
        st.sidebar.subheader("Home Score")
        home_scores = filtered_df['Home Score'].dropna()
        if len(home_scores) > 0:
            # Convert to numeric and remove any non-numeric values
            home_scores_numeric = pd.to_numeric(home_scores, errors='coerce').dropna()
            if len(home_scores_numeric) > 0:
                min_home = int(home_scores_numeric.min())
                max_home = int(home_scores_numeric.max())
                
                # Only show slider if min != max
                if min_home != max_home:
                    selected_home_range = st.sidebar.slider(
                        "Home Score Range:",
                        min_value=min_home,
                        max_value=max_home,
                        value=(min_home, max_home),
                        key="home_score_filter"
                    )
                    filtered_df = filtered_df[
                        (pd.to_numeric(filtered_df['Home Score'], errors='coerce') >= selected_home_range[0]) & 
                        (pd.to_numeric(filtered_df['Home Score'], errors='coerce') <= selected_home_range[1])
                    ]
                else:
                    st.sidebar.write(f"All Home Scores: {min_home}")
            else:
                st.sidebar.write("No valid Home Score data")
    
    # Away Score filter
    if 'Away Score' in df.columns:
        st.sidebar.subheader("Away Score")
        away_scores = filtered_df['Away Score'].dropna()
        if len(away_scores) > 0:
            # Convert to numeric and remove any non-numeric values
            away_scores_numeric = pd.to_numeric(away_scores, errors='coerce').dropna()
            if len(away_scores_numeric) > 0:
                min_away = int(away_scores_numeric.min())
                max_away = int(away_scores_numeric.max())
                
                # Only show slider if min != max
                if min_away != max_away:
                    selected_away_range = st.sidebar.slider(
                        "Away Score Range:",
                        min_value=min_away,
                        max_value=max_away,
                        value=(min_away, max_away),
                        key="away_score_filter"
                    )
                    filtered_df = filtered_df[
                        (pd.to_numeric(filtered_df['Away Score'], errors='coerce') >= selected_away_range[0]) & 
                        (pd.to_numeric(filtered_df['Away Score'], errors='coerce') <= selected_away_range[1])
                    ]
                else:
                    st.sidebar.write(f"All Away Scores: {min_away}")
            else:
                st.sidebar.write("No valid Away Score data")
    
    # Display filtered data info
    st.sidebar.markdown("---")
    st.sidebar.metric("Filtered Records", len(filtered_df))
    st.sidebar.metric("Total Records", len(df))
    
    # Main content area
    if len(filtered_df) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Odds Difference vs Time Stamp")
            
            # Create scatter plot using matplotlib
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Filter out rows without combined time data
            plot_data = filtered_df.dropna(subset=['Combined_Time', 'V2_minus_V1'])
            
            if len(plot_data) > 0:
                # Create color mapping based on V2_minus_V1 difference levels
                v2_diff = plot_data['V2_minus_V1']
                
                # Define difference level bins
                diff_bins = [-np.inf, -0.1, -0.05, 0, 0.05, 0.1, np.inf]
                diff_labels = ['Large Negative (<-0.1)', 'Moderate Negative (-0.1 to -0.05)', 
                              'Small Negative (-0.05 to 0)', 'Small Positive (0 to 0.05)',
                              'Moderate Positive (0.05 to 0.1)', 'Large Positive (>0.1)']
                
                # Assign colors to each bin
                diff_colors = ['darkred', 'red', 'lightcoral', 'lightgreen', 'green', 'darkgreen']
                
                # Create bins and assign colors
                plot_data = plot_data.copy()
                plot_data['Diff_Level'] = pd.cut(v2_diff, bins=diff_bins, labels=diff_labels, include_lowest=True)
                
                # Plot each difference level with different colors
                for i, level in enumerate(diff_labels):
                    level_data = plot_data[plot_data['Diff_Level'] == level]
                    if len(level_data) > 0:
                        scatter = ax.scatter(
                            level_data['Combined_Time'], 
                            level_data['V2_Clean_Selection'],  # Use clean selection names
                            c=diff_colors[i], 
                            s=60,  # Fixed size for better visibility
                            alpha=0.7,
                            label=level,
                            edgecolors='black',
                            linewidth=0.5
                        )
                
                # Customize x-axis to show quarter labels
                if len(plot_data) > 0:
                    min_time = plot_data['Combined_Time'].min()
                    max_time = plot_data['Combined_Time'].max()
                    
                    # Debug: Show what quarters are available
                    available_quarters = plot_data['Quarter'].dropna().unique()
                    st.write(f"Debug - Available quarters: {available_quarters}")
                    st.write(f"Debug - Time range: {min_time} to {max_time}")
                    
                    # Create quarter markers based on actual data quarters
                    quarter_ticks = []
                    quarter_labels = []
                    
                    # For each quarter in the data, find its representative time position
                    for quarter in sorted(available_quarters, key=lambda x: (len(x), x)):
                        if quarter.startswith('Q'):
                            # Get data points for this quarter
                            quarter_data = plot_data[plot_data['Quarter'] == quarter]
                            if len(quarter_data) > 0:
                                # Use the middle time point of this quarter's data
                                quarter_time = quarter_data['Combined_Time'].mean()
                                quarter_ticks.append(quarter_time)
                                quarter_labels.append(quarter)
                    
                    st.write(f"Debug - Quarter ticks: {quarter_ticks}")
                    st.write(f"Debug - Quarter labels: {quarter_labels}")
                    
                    if quarter_ticks:
                        ax.set_xticks(quarter_ticks)
                        ax.set_xticklabels(quarter_labels)
                
                ax.set_xlabel('Game Time (Quarters)')
                ax.set_ylabel('V2 Selection (Clean)')
                ax.set_title('V2 Selection vs Game Time (Colored by V2-V1 Difference Level)')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("No data available with valid timestamp information for plotting.")
            
            # Second scatter plot: V1 and V2 vs Game Time
            st.subheader("📈 V1 vs V2 Values Over Time")
            
            # Filter out rows without combined time data
            values_plot_data = filtered_df.dropna(subset=['Combined_Time', 'V1', 'V2'])
            
            if len(values_plot_data) > 0:
                fig_values, ax_values = plt.subplots(figsize=(14, 8))
                
                # Plot V1 Values
                scatter_v1 = ax_values.scatter(
                    values_plot_data['Combined_Time'], 
                    values_plot_data['V1'],
                    c='blue', 
                    s=40,
                    alpha=0.6,
                    label='V1 Values',
                    edgecolors='black',
                    linewidth=0.3
                )
                
                # Plot V2 Values
                scatter_v2 = ax_values.scatter(
                    values_plot_data['Combined_Time'], 
                    values_plot_data['V2'],
                    c='red', 
                    s=40,
                    alpha=0.6,
                    label='V2 Values',
                    edgecolors='black',
                    linewidth=0.3
                )
                
                # Add labels for V2 Clean Selection (only show a subset to avoid overcrowding)
                if len(values_plot_data) <= 50:  # Show labels only if not too many points
                    for idx, row in values_plot_data.iterrows():
                        if pd.notna(row['V2_Clean_Selection']):
                            clean_label = str(row['V2_Clean_Selection'])
                            # Truncate label if too long
                            if len(clean_label) > 25:
                                clean_label = clean_label[:25] + '...'
                            
                            # Add label for V1 point
                            ax_values.annotate(
                                clean_label,
                                (row['Combined_Time'], row['V1']),
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=8, alpha=0.7, color='blue'
                            )
                            # Add label for V2 point
                            ax_values.annotate(
                                clean_label,
                                (row['Combined_Time'], row['V2']),
                                xytext=(5, -15), textcoords='offset points',
                                fontsize=8, alpha=0.7, color='red'
                            )
                else:
                    st.info("Too many data points to show labels clearly. Labels are hidden for better visualization.")
                
                # Customize x-axis to show quarter labels
                min_time = values_plot_data['Combined_Time'].min()
                max_time = values_plot_data['Combined_Time'].max()
                
                # Create quarter markers based on actual data quarters
                available_quarters = values_plot_data['Quarter'].dropna().unique()
                quarter_ticks = []
                quarter_labels = []
                
                # For each quarter in the data, find its representative time position
                for quarter in sorted(available_quarters, key=lambda x: (len(x), x)):
                    if quarter.startswith('Q'):
                        # Get data points for this quarter
                        quarter_data = values_plot_data[values_plot_data['Quarter'] == quarter]
                        if len(quarter_data) > 0:
                            # Use the middle time point of this quarter's data
                            quarter_time = quarter_data['Combined_Time'].mean()
                            quarter_ticks.append(quarter_time)
                            quarter_labels.append(quarter)
                
                if quarter_ticks:
                    ax_values.set_xticks(quarter_ticks)
                    ax_values.set_xticklabels(quarter_labels)
                
                ax_values.set_xlabel('Game Time (Quarters)')
                ax_values.set_ylabel('V1 and V2 Values')
                ax_values.set_title('V1 Values (Blue) vs V2 Values (Red) Over Game Time')
                ax_values.legend()
                ax_values.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig_values)
            else:
                st.warning("No data available with valid timestamp and V1/V2 values for plotting.")
            
            # Trend analysis
            st.subheader("📈 Trend Analysis")
            
            # Create trend charts
            trend_col1, trend_col2 = st.columns(2)
            
            with trend_col1:
                # Distribution of V2 - V1 differences
                fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
                ax_hist.hist(filtered_df['V2_minus_V1'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax_hist.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='No Difference')
                ax_hist.set_xlabel('V2 - V1 Difference')
                ax_hist.set_ylabel('Frequency')
                ax_hist.set_title('Distribution of V2 - V1 Differences')
                ax_hist.legend()
                ax_hist.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig_hist)
            
            with trend_col2:
                # Time-based analysis
                if 'Combined_Time' in filtered_df.columns and filtered_df['Combined_Time'].notna().any():
                    try:
                        fig_time, ax_time = plt.subplots(figsize=(8, 6))
                        
                        # Group by quarter and calculate mean difference
                        quarter_data = filtered_df.dropna(subset=['Quarter', 'V2_minus_V1']).groupby('Quarter')['V2_minus_V1'].agg(['mean', 'count']).reset_index()
                        
                        if len(quarter_data) > 0:
                            bars = ax_time.bar(quarter_data['Quarter'], quarter_data['mean'], 
                                              alpha=0.7, color='lightblue', edgecolor='black')
                            ax_time.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                            ax_time.set_xlabel('Quarter')
                            ax_time.set_ylabel('Average V2 - V1 Difference')
                            ax_time.set_title('Average V2 - V1 Difference by Quarter')
                            ax_time.grid(True, alpha=0.3)
                            
                            # Add count labels on bars
                            for bar, count in zip(bars, quarter_data['count']):
                                height = bar.get_height()
                                ax_time.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                           f'n={count}', ha='center', va='bottom', fontsize=8)
                            
                            plt.tight_layout()
                            st.pyplot(fig_time)
                            plt.close(fig_time)  # Close figure to free memory
                        else:
                            st.info("No quarter data available for analysis")
                    except Exception as e:
                        st.error(f"Error creating time-based chart: {str(e)}")
                        st.info("Showing alternative analysis instead")
                        
                        # Fallback to simple text analysis
                        if 'Quarter' in filtered_df.columns:
                            quarter_summary = filtered_df.groupby('Quarter')['V2_minus_V1'].agg(['mean', 'count']).round(3)
                            st.write("**Quarter Summary:**")
                            st.dataframe(quarter_summary)
                else:
                    # Box plot by market as fallback
                    if len(filtered_df['Market'].unique()) > 1:
                        try:
                            fig_box, ax_box = plt.subplots(figsize=(8, 6))
                            
                            # Create box plot manually using matplotlib
                            markets = filtered_df['Market'].unique()
                            market_data = []
                            market_labels = []
                            
                            for market in markets:
                                market_odds = filtered_df[filtered_df['Market'] == market]['V2_minus_V1'].dropna()
                                if len(market_odds) > 0:  # Only add if there's data
                                    market_data.append(market_odds)
                                    market_labels.append(market)
                            
                            if market_data:  # Only create plot if there's data
                                bp = ax_box.boxplot(market_data, labels=market_labels, patch_artist=True)
                                
                                # Color the boxes
                                colors = plt.cm.Set3(np.linspace(0, 1, len(market_data)))
                                for patch, color in zip(bp['boxes'], colors):
                                    patch.set_facecolor(color)
                                    patch.set_alpha(0.7)
                                
                                ax_box.set_title('V2 - V1 Difference by Market')
                                ax_box.set_ylabel('V2 - V1 Difference')
                                ax_box.tick_params(axis='x', rotation=45)
                                ax_box.grid(True, alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig_box)
                                plt.close(fig_box)  # Close figure to free memory
                            else:
                                st.info("No market data available for comparison")
                        except Exception as e:
                            st.error(f"Error creating box plot: {str(e)}")
                    else:
                        st.info("Need multiple markets for comparison")
        
        with col2:
            st.subheader("📋 Summary Statistics")
            
            # Key metrics
            v2_minus_v1 = filtered_df['V2_minus_V1']
            
            st.metric("Mean V2 - V1 Difference", f"{v2_minus_v1.mean():.3f}")
            st.metric("Median V2 - V1 Difference", f"{v2_minus_v1.median():.3f}")
            st.metric("Std Deviation", f"{v2_minus_v1.std():.3f}")
            
            st.markdown("---")
            
            # Positive vs Negative differences
            positive_diff = (v2_minus_v1 > 0).sum()
            negative_diff = (v2_minus_v1 < 0).sum()
            neutral_diff = (v2_minus_v1 == 0).sum()
            
            st.write("**V2 - V1 Difference Breakdown:**")
            st.write(f"• Positive (V2 > V1): {positive_diff} ({positive_diff/len(v2_minus_v1)*100:.1f}%)")
            st.write(f"• Negative (V2 < V1): {negative_diff} ({negative_diff/len(v2_minus_v1)*100:.1f}%)")
            st.write(f"• Neutral (V2 = V1): {neutral_diff} ({neutral_diff/len(v2_minus_v1)*100:.1f}%)")
            
            st.markdown("---")
            
            # Top selections by V2 - V1 difference
            st.subheader("🔝 Extreme V2 - V1 Differences")
            
            # Highest positive differences
            top_positive = filtered_df.nlargest(10, 'V2_minus_V1')[['V2 Selection', 'V2_minus_V1', 'Market']]
            st.write("**Highest Positive Differences:**")
            for _, row in top_positive.iterrows():
                st.write(f"• {row['Market']} - {row['V2 Selection']}: +{row['V2_minus_V1']:.3f}")
            
            # Lowest negative differences
            top_negative = filtered_df.nsmallest(10, 'V2_minus_V1')[['V2 Selection', 'V2_minus_V1', 'Market']]
            st.write("**Lowest Negative Differences:**")
            for _, row in top_negative.iterrows():
                st.write(f"• {row['Market']} - {row['V2 Selection']}: {row['V2_minus_V1']:.3f}")
        
        # Data table
        st.subheader("📄 Filtered Data")
        
        # Display options
        show_all_columns = st.checkbox("Show all columns", value=False)
        
        if show_all_columns:
            display_df = filtered_df
        else:
            # Show key columns
            key_columns = ['Market', 'V2 Selection', 'V2_Clean_Selection', 'V2_Player', 'V2_Line', 'Quarter', 'Time_Minutes', 'V1', 'V2', 
                          'V2_minus_V1', 'V1 Odds', 'V2 Odds', 'Timestamp', 'Home Score', 'Away Score']
            available_key_columns = [col for col in key_columns if col in filtered_df.columns]
            display_df = filtered_df[available_key_columns]
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=300
        )
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_betting_odds.csv",
            mime="text/csv"
        )
    
    else:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
    
    # Show sample data format
    st.subheader("Expected Data Format")
    sample_data = {
        'Market': ['(2) 1ST QUARTER MONEY LINE', '(4) 3RD QUARTER MONEY LINE'],
        'V2 Selection': ['(3) Philadelphia Eagles', '(9) Philadelphia Eagles'],
        'V1 Odds': [2.11, 1.48],
        'V2 Odds': [1.88, 1.68],
        'Timestamp': ['Q0 - 15:00', 'Q0 - 15:00'],
        'Home Score': [0, 0],
        'Away Score': [0, 0]
    }
    st.dataframe(pd.DataFrame(sample_data))
