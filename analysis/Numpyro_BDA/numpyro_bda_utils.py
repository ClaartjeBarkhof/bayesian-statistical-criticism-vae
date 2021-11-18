def predictive_checks(model, prior=False):
    if prior:
        if model.prior_predictions is None:
            model.get_prior_predictions()
        preds = np.array(model.prior_predictions["obs"])
    else:
        if model.posterior_predictions is None:
            model.get_posterior_predictions()
        preds = np.array(model.posterior_predictions["obs"]) #.reshape(-1, self.posterior_predictions["obs"].shape[-1])
        preds = preds.reshape(model.num_chains, model.num_samples, model.N)
        preds = preds[0, :, :]  # only consider the first chain, to make plotting less heavy

    # print("predictive checks shape preds:", preds.shape)

    # SHAPES:
    # posterior_predictions [N_chains*N_samples, N_data]
    # prior_predictions [N_samples, N_data]
    # samples (generalising both): [N_s, N_d]

    obs_mean = np.mean(model.obs_y)
    obs_std = np.std(model.obs_y)
    obs_median = np.median(model.obs_y)
    obs_mode = np.max(model.obs_y)
    obs_skew = obs_mean ** (-0.5)
    obs_kurtosis = obs_mean ** (-1)

    obs_stats = [obs_mean, obs_std, obs_median, obs_mode, obs_skew, obs_kurtosis]

    pred_mean = np.mean(preds, axis=1)
    pred_std = np.std(preds, axis=1)
    pred_median = np.median(preds, axis=1)
    pred_mode = np.max(preds, axis=1)
    pred_skew = pred_mean ** (-0.5)
    pred_kurtosis = pred_mean ** (-1)

    preds_stats = [pred_mean, pred_std, pred_median, pred_mode, pred_skew, pred_kurtosis]
    preds_stats_means = [s.mean() for s in preds_stats]
    preds_stats_std = [s.std() for s in preds_stats]

    # predictive p values
    p_vals = [(p > o).mean() for p, o in zip(preds_stats, obs_stats)]

    stats = ["mean", "std", "median", "max", "skew", "kurtosis"]

    headers = ['check', 'p_val', 'obs', 'pred (mean)', 'pred (std)']
    rows = [['S', 1, None, preds.shape[0], None], ['shape', None, model.obs_y.shape, preds.shape, None]]

    for s, p_v, o, p_m, p_std in zip(stats, p_vals, obs_stats, preds_stats_means, preds_stats_std):
        rows.append([s, f"{p_v:.3f}", f"{o:.3f}", f"{p_m:.3f}", f"{p_std:.3f}"])

    for C in [0.25, 0.5, 0.75, 1., 2.]:
        mean_check = (np.abs(pred_mean - obs_mean) < C * pred_std)
        rows.append([f"mean within {C:.2f}*std", None, None, f"{mean_check.mean():.3f}", f"{mean_check.std():.3f}"])

    print(tabulate(rows, headers=headers))  # , floatfmt=(None, ".3f", ".3f", ".3f")

    ncols = 3
    nrows = int(np.ceil(len(stats) / ncols))

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4))

    for i, (s, o, p, pm) in enumerate(zip(stats, obs_stats, preds_stats, preds_stats_means)):
        r, c = i // ncols, i % ncols
        if np.any(np.isinf(p)) or np.any(np.isnan(p)):
            print(f"encountered NAN in preds of {s} = {np.any(np.isnan(p))}")
            print(f"encountered INF in preds of {s} = {np.any(np.isinf(p))}")
            continue

        axs[r, c].hist(np.array(p), bins=40, lw=0, density=True)
        axs[r, c].axvline(o, color='g', linestyle='dashed', label='obs')
        axs[r, c].axvline(pm, color='r', linestyle='dashed', label='pred mean T')
        axs[r, c].set_title(s)

    title = "Prior predictive checks" if prior else "Posterior predictive checkes"
    plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_group_df(df, group_by, group_id_col="group_id", group_name_col="group_name"):
    df[group_id_col] = df.groupby(group_by).grouper.group_info[0]
    df[group_name_col] = df[group_by].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    group_df = df[group_by + [group_id_col]].value_counts().reset_index()[group_by + [group_id_col]]
    group_df[group_name_col] = group_df[group_by].apply(lambda row: ' & '.join(row.values.astype(str)),
                                                        axis=1).values
    return df, group_df