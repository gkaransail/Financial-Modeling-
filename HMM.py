def detect_normal_states_verbose(group, n_states=2):
    results = {}
    
    for col in ["Unit Cost smooth", "margin smooth"]:
        series = group[col].dropna().values.reshape(-1, 1)
        print(f"\nProcessing column: {col}")
        print(f"Original series (first 10 rows):\n{series[:10]}")
        
        # Skip groups with not enough data
        if len(series) < n_states:
            results[f"Normalized_{col}"] = np.nan
            results[f"{col}_std"] = np.nan
            print("Not enough data to fit HMM. Skipping this group.")
            continue
        
        # Fit HMM
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
        model.fit(series)
        print("HMM fitted.")
        
        # Predict regimes
        hidden_states = model.predict(series)
        print(f"Predicted hidden states (first 20 rows): {hidden_states[:20]}")
        
        # Calculate regime means
        regime_means = [series[hidden_states == i].mean() for i in range(n_states)]
        print(f"Regime means: {regime_means}")
        overall_mean = series.mean()
        print(f"Overall mean: {overall_mean}")
        
        # Choose regime closest to overall mean as "normal"
        normal_state = np.argmin(np.abs(np.array(regime_means) - overall_mean))
        normalized_value = regime_means[normal_state]
        print(f"Selected normal state: {normal_state}, Normalized value: {normalized_value}")
        
        # Standard deviation of the "normal" regime
        normal_std = series[hidden_states == normal_state].std()
        print(f"Standard deviation in normal regime: {normal_std}")
        
        results[f"Normalized_{col}"] = normalized_value
        results[f"{col}_std"] = normal_std

    return pd.Series(results)
