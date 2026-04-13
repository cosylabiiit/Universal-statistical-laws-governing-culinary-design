from matplotlib import ticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, lognorm, weibull_min, gamma, expon, uniform, ks_2samp, gaussian_kde


class NutrientPlotter:
    tick_size = 12
    label_size = 12
    legend_size = 12
    
    NUTRIENT_COLORS = {
        'Carbohydrate, by difference (g)': '#d73027',
        'Protein (g)': '#58ff87',
        'Total lipid (fat) (g)': '#4575b4',
    }
    
    def __init__(self, nm):
        self.nm = nm
    
    def _get_nutrient_colors(self):
        stats = self.nm.all_stats()
        df_sorted = stats.sort_values('mean').reset_index(drop=True)
        colors = [self.NUTRIENT_COLORS.get(n, '#999999') for n in df_sorted['nutrient']]
        return df_sorted, colors
    
    def plot_Qxn(self, nutrients_dict, nbins=15, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        
        n = len(nutrients_dict)
        all_x_min, all_x_max, all_y_max = [], [], []
        
        for i, (nutrient, props) in enumerate(nutrients_dict.items()):
            hist = self.nm.Qxn(nutrient, nbins=nbins)
            log_centers = np.log10(hist['centers'])
            if len(log_centers) > 1:
                width = np.diff(log_centers)[0] * 0.7 / n
            else:
                width = 0.1
            
            offset = 10 ** (width * (i - n/2 + 0.5))
            centers_offset = hist['centers'] * offset
            
            ax.bar(centers_offset, hist['Q'], width=hist['centers'] * (10**width - 1), color=props['color'], alpha=0.7, label=props['label'], edgecolor='black', linewidth=0.5)
            ax.plot(hist['x_fit'], hist['Q_fit'], color=props['color'], lw=2.5, alpha=0.9, linestyle='--')
            
            all_x_min.append(hist['centers'].min())
            all_x_max.append(hist['centers'].max())
            all_y_max.append(max(hist['Q'].max(), hist['Q_fit'].max()))
        
        x_min = min(all_x_min) * 0.5
        x_max = max(all_x_max) * 2
        y_max = max(all_y_max) * 1.15
        
        y_max_rounded = np.ceil(y_max / 0.05) * 0.05
        
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max_rounded)
        ax.set_xlabel(r"$\mathbf{x_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"$\mathbf{Q(x_n)}$", fontsize=self.label_size, fontweight='bold')

        ax.set_yticks(np.arange(0, y_max_rounded + 0.01, 0.05))
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        ax.legend(loc='upper right', framealpha=0.9, fontsize=self.legend_size)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_log_std(self, figsize=(12, 8)):
        # Scatter plot of log-std vs mean
        fig, ax = plt.subplots(figsize=figsize)
        df_sorted, colors = self._get_nutrient_colors()
        
        x_data = df_sorted["mean"].values
        y_data = df_sorted["log_std"].values
        
        ax.scatter(x_data, y_data, c=colors, s=120, alpha=0.8, edgecolors='black', linewidths=1, zorder=3)
        
        summary = self.nm.log_std_summary()
        ax.axhspan(summary['mean'] - summary['std'], summary['mean'] + summary['std'], color='gray', alpha=0.2, label=rf'$\langle s_n \rangle \pm \sigma$')
        ax.axhline(y=summary['mean'], color='gray', linestyle='--', linewidth=2)
        
        for idx, row in df_sorted.iterrows():
            nutrient = row['nutrient']
            label = nutrient.replace('Content', '')
            x = row['mean']
            y = row['log_std']
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=self.label_size, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7), fontweight='bold')
        
        ax.set_xlabel(r"Mean concentration, $\mathbf{\mu_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"Log standard deviation, $\mathbf{s_n}$", fontsize=self.label_size, fontweight='bold')

        ax.set_xscale('log')
        x_min, x_max = x_data.min(), x_data.max()
        ax.set_xlim(x_min * 0.3, x_max * 3)

        ax.set_ylim(0, 3.0)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        plt.tight_layout()
        
        return fig, ax, summary

    def plot_raw_std(self, figsize=(12, 8)):
        # Scatter plot of raw std vs mean
        fig, ax = plt.subplots(figsize=figsize)
        df_sorted, colors = self._get_nutrient_colors()
        
        x_data = df_sorted["mean"].values
        y_data = df_sorted["std"].values
        
        ax.scatter(x_data, y_data, c=colors, s=120, alpha=0.8, edgecolors='black', linewidths=1, zorder=3)

        summary = self.nm.std_summary()
        ax.axhspan(summary['mean'] - summary['std'], summary['mean'] + summary['std'], color='gray', alpha=0.2, label=rf'$\langle s_n \rangle \pm \sigma$')
        ax.axhline(y=summary['mean'], color='gray', linestyle='--', linewidth=2)
        
        for idx, row in df_sorted.iterrows():
            nutrient = row['nutrient']
            label = nutrient.replace('Content', '')
            x = row['mean']
            y = row['std']
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=self.label_size,clip_on=True, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7), fontweight='bold')
        
        ax.set_xlabel(r"Mean concentration, $\mathbf{\mu_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"Standard deviation, $\mathbf{s_n}$", fontsize=self.label_size, fontweight='bold')

        x_min, x_max = x_data.min(), x_data.max()
        ax.set_xlim(x_min * 0.3, x_max * 3)

        y_min, y_max = y_data.min(), y_data.max()
        ax.set_ylim(0, y_max * 1.5)
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        plt.tight_layout()
        
        return fig, ax

    
    def plot_log_skew(self, figsize=(12, 8)):
        # Scatter plot of log-skewness vs mean
        fig, ax = plt.subplots(figsize=figsize)
        
        df_sorted, colors = self._get_nutrient_colors()
        
        x_data = df_sorted['mean'].values
        y_data = df_sorted['log_skew'].values
        
        ax.scatter(x_data, y_data, c=colors, s=120, alpha=0.8, edgecolors='black', linewidths=1, zorder=3)
        
        summary = self.nm.log_skew_summary()
        ax.axhline(y=summary['mean'], color='gray', linestyle='--', linewidth=2, label=f'Mean $\\langle sk_n \\rangle = {summary["mean"]:.2f}$')
        ax.axhline(y=0, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Perfect symmetry (sk=0)', zorder=2)
        
        for idx, row in df_sorted.iterrows():
            nutrient = row['nutrient']
            label = nutrient.replace('Content', '')
            x = row['mean']
            y = row['log_skew']
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=self.label_size, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7), fontweight='bold')
        
        ax.set_xlabel(r"Mean concentration, $\mathbf{\mu_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"Log-skewness, $\mathbf{sk_n}$", fontsize=self.label_size, fontweight='bold')

        ax.set_xscale('log')
        x_min, x_max = x_data.min(), x_data.max()
        ax.set_xlim(x_min * 0.3, x_max * 3)
        
        ax.set_ylim(-2, 2)
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=self.legend_size)
        plt.tight_layout()
        
        return fig, ax, summary
    
    def plot_rescaled_Q(self, nutrients, nbins=15, figsize=(10, 8)):
        # Plot rescaled Q(y_n) showing translational invariance
        fig, ax = plt.subplots(figsize=figsize)
        records = []
        
        for i, name in enumerate(nutrients):
            res = self.nm.rescaled_Q(name, nbins=nbins)
            color = self.NUTRIENT_COLORS.get(name, '#999999')
            
            short_label = name.replace(' (g)', '').replace(', by difference', '')
            
            mu_fit, sigma_fit = norm.fit(res['log_y'])
            x_fit = np.linspace(-8, 8, 1000)
            ax.plot(x_fit, norm.pdf(x_fit, mu_fit, sigma_fit), '--', color=color, lw=2.5, alpha=0.8, label=f'{short_label} fit (σ={sigma_fit:.2f})')
            ax.scatter(res['centers'], res['Q'], s=90, alpha=0.85, color=color, edgecolors='white', linewidths=1.5, label=f'{short_label} data', zorder=3)
            
            records.append({
                'name': name, 'm_n': res['m_n'], 's_n': res['s_n'],
                'translated_mean': np.mean(res['log_y']),
                'translated_std': np.std(res['log_y'], ddof=1),
            })
        
        ax.set_xlabel(r'$\mathbf{log(y_n)}$', fontsize=self.tick_size, fontweight='bold')
        ax.set_ylabel(r'$\mathbf{Q(y_n)}$', fontsize=self.tick_size, fontweight='bold')
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(0, 0.35)
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=self.label_size, ncol=2, edgecolor='#cccccc', fancybox=True)
        plt.tight_layout()
        
        return fig, ax, pd.DataFrame(records)
    
    def plot_ks_distance_cdf(self, ks_df, figsize=(10, 7)):
        # Plot smoothed CDF of K-S distances for each distribution
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        distributions = {
            'D_lognormal':   {'label': 'Lognormal',    'color': '#1f77b4'},
            'D_weibull':     {'label': 'Weibull',      'color': '#ff7f0e'},
            'D_gamma':       {'label': 'Gamma',        'color': '#2ca02c'},
            'D_gaussian':    {'label': 'Gaussian',     'color': '#9467bd'},
            'D_exponential': {'label': 'Exponential',  'color': '#d62728'},
            'D_uniform':     {'label': 'Uniform',      'color': '#17becf'},
        }   
        
        # Find global minimum D for x-axis
        all_d_min = 1.0
        for col in distributions.keys():
            if col in ks_df.columns:
                d_vals = ks_df[col].dropna().values
                if len(d_vals) > 0:
                    all_d_min = min(all_d_min, d_vals.min())
        
        # Reference lines
        ax.axvline(x=0.1, color='#333333', linestyle='--', lw=2.5, alpha=0.9, zorder=1)
        ax.axvline(x=0.2, color='#333333', linestyle=':', lw=2.5, alpha=0.9, zorder=1)
        ax.text(0.1, 1.02, 'D=0.1', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')
        ax.text(0.2, 1.02, 'D=0.2', ha='center', va='bottom', fontsize=12, fontweight='bold', color='#333333')
        
        for col, props in distributions.items():
            if col not in ks_df.columns:
                continue
            d_values = ks_df[col].dropna().values
            if len(d_values) < 2:
                continue
            
            kde = gaussian_kde(d_values, bw_method='scott')
            x_smooth = np.linspace(all_d_min * 0.9, 1, 200)
            cdf_smooth = np.array([kde.integrate_box_1d(all_d_min * 0.9, x) for x in x_smooth])
            cdf_smooth = cdf_smooth / cdf_smooth[-1] if cdf_smooth[-1] > 0 else cdf_smooth
            
            ax.plot(x_smooth, cdf_smooth, color=props['color'], lw=2.5, linestyle='-', 
                    label=props['label'], alpha=0.85, zorder=2)
        
        ax.set_xlabel('Kolmogorov-Smirnov Distance', fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=self.label_size, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.08)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        ax.legend(loc='lower right', fontsize=self.legend_size, framealpha=0.95, edgecolor='#cccccc', fancybox=True)
        ax.grid(False)
        plt.tight_layout()
        
        return fig, ax
    
    def _sample_lognormal(self, nutrient, n_samples=10000):
        shape, loc, scale = self.nm.ks_test(nutrient)['lognormal']['params']
        samples = lognorm.rvs(shape, loc=loc, scale=scale, size=n_samples)
        samples = samples[samples > 0]
        return samples.mean(), samples.std()
    
    def _sample_weibull(self, nutrient, n_samples=10000):
        c, loc, scale = self.nm.ks_test(nutrient)['weibull']['params']
        samples = weibull_min.rvs(c, loc=loc, scale=scale, size=n_samples)
        samples = samples[samples > 0]
        return samples.mean(), samples.std()
    
    def _sample_gamma(self, nutrient, n_samples=10000):
        a, loc, scale = self.nm.ks_test(nutrient)['gamma']['params']
        samples = gamma.rvs(a, loc=loc, scale=scale, size=n_samples)
        samples = samples[samples > 0]
        return samples.mean(), samples.std()
    
    def _sample_gaussian(self, nutrient, n_samples=10000):
        mu_fit, sigma_fit = self.nm.ks_test(nutrient)['gaussian']['params']
        samples = np.abs(norm.rvs(loc=mu_fit, scale=sigma_fit, size=n_samples))
        return samples.mean(), samples.std()
    
    def _sample_exponential(self, nutrient, n_samples=10000):
        loc_e, scale_e = self.nm.ks_test(nutrient)['exponential']['params']
        samples = expon.rvs(loc=loc_e, scale=scale_e, size=n_samples)
        return samples.mean(), samples.std()
    
    def _sample_uniform(self, nutrient, n_samples=10000):
        loc_u, scale_u = self.nm.ks_test(nutrient)['uniform']['params']
        samples = uniform.rvs(loc=loc_u, scale=scale_u, size=n_samples)
        return samples.mean(), samples.std()

    def compare_cv_distributions(self, n_samples=10000):
        sample_funcs = {
            'lognormal': self._sample_lognormal,
            'weibull': self._sample_weibull,
            'gamma': self._sample_gamma,
            'gaussian': self._sample_gaussian,
            'exponential': self._sample_exponential,
            'uniform': self._sample_uniform,
        }
        
        stats_df = self.nm.all_stats()
        cv_real = stats_df['std'].values / stats_df['mean'].values
        
        results = []
        for dist, func in sample_funcs.items():
            mu_sim, sigma_sim = [], []
            for nutrient in self.nm.nutrients:
                try:
                    mu, sigma = func(nutrient, n_samples)
                    mu_sim.append(mu)
                    sigma_sim.append(sigma)
                except:
                    mu_sim.append(np.nan)
                    sigma_sim.append(np.nan)
            
            cv_sim = np.array(sigma_sim) / np.array(mu_sim)
            valid_cv = np.isfinite(cv_real) & np.isfinite(cv_sim) & (cv_real > 0) & (cv_sim > 0)
            
            ks_stat, ks_p = ks_2samp(cv_real[valid_cv], cv_sim[valid_cv])
            spread_diff = abs(np.std(np.log(cv_real[valid_cv])) - np.std(np.log(cv_sim[valid_cv])))
            
            results.append({
                'distribution': dist,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_p,
                'spread_diff': spread_diff,
            })
        
        return pd.DataFrame(results).sort_values('spread_diff').reset_index(drop=True)
    
    def plot_sigma_vs_mu(self, figsize=(10, 8)):
        # Plot sigma vs mu with power-law fit (Taylor's law)
        fig, ax = plt.subplots(figsize=figsize)

        nutrient_stats = self.nm.all_stats()
        df_sorted = nutrient_stats.sort_values('mean').reset_index(drop=True)

        # Plot each nutrient with fixed colors
        for i, (idx, row) in enumerate(df_sorted.iterrows()):
            color = self.NUTRIENT_COLORS.get(row['nutrient'], '#999999')
            short_label = row['nutrient'].replace(' (g)', '').replace(', by difference', '')
            ax.scatter(row['mean'], row['std'], c=[color], s=120, alpha=0.9, edgecolors='black', linewidths=1, zorder=3, label=short_label)
            ax.annotate(short_label, (row['mean'], row['std']), textcoords="offset points", xytext=(8, 8), fontsize=self.label_size, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

        valid_idx = (df_sorted['mean'] > 0) & (df_sorted['std'] > 0)
        x_fit = np.log(df_sorted.loc[valid_idx, 'mean'])
        y_fit = np.log(df_sorted.loc[valid_idx, 'std'])
        valid_idx2 = np.isfinite(x_fit) & np.isfinite(y_fit)
        x_fit, y_fit = x_fit[valid_idx2], y_fit[valid_idx2]

        slope, intercept = np.polyfit(x_fit, y_fit, 1)
        R_squared = np.corrcoef(x_fit, y_fit)[0, 1]**2

        x_min, x_max = df_sorted['mean'].min(), df_sorted['mean'].max()
        x_line = np.logspace(np.log10(x_min * 0.1), np.log10(x_max * 10), 200)
        ax.plot(x_line, np.exp(intercept) * x_line**slope, 'r-', lw=3, alpha=0.8, label=rf'Power-law fit: $\sigma \propto \mu^{{{slope:.2f}}}$  ($R^2$={R_squared:.3f})')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(r"Mean concentration, $\mathbf{\mu_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"Standard deviation, $\mathbf{\sigma_n}$", fontsize=self.label_size, fontweight='bold')

        ax.set_xlim(x_min * 0.3, x_max * 5)
        y_min, y_max = df_sorted['std'].min(), df_sorted['std'].max()
        ax.set_ylim(y_min * 0.3, y_max * 5)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        ax.legend(fontsize=self.legend_size, loc='upper left', framealpha=0.95, edgecolor='gray')
        ax.grid(False)
        plt.tight_layout()

        return fig, ax, {'slope': slope, 'intercept': intercept, 'R_squared': R_squared}

