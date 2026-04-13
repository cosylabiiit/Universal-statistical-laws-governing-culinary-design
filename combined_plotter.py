import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm, lognorm, weibull_min, gamma, expon, uniform, gaussian_kde
import os


class CombinedPlotter:
    tick_size = 12
    label_size = 12
    legend_size = 12
    
    CUISINE_COLORS = {
        "Italian":    "#e41a1c",  # Red
        "Mexican":    "#377eb8",  # Blue
        "South American": "#ff7f00",  # Orange
        "Canadian":   "#4daf4a",  # Green
        "Indian Subcontinent": "#ffff33",  # Yellow
        "French":     "#984ea3",  # Purple
        "Chinese and Mongolian": "#a65628",  # Brown
        "US":         "#f781bf",  # Pink
        "Australian": "#999999",  # Gray
        "UK":         "#66c2a5",  # Teal
    }
    
    NUTRIENT_MARKERS = {
        "Carbohydrate, by difference (g)": "o",   # Circle
        "Protein (g)": "s",                        # Square
        "Total lipid (fat) (g)": "^",              # Triangle
    }
    
    NUTRIENT_LINESTYLES = {
        "Carbohydrate, by difference (g)": "-",    # Solid
        "Protein (g)": "--",                       # Dashed
        "Total lipid (fat) (g)": ":",             # Dotted
    }
    
    NUTRIENT_SHORT = {
        "Carbohydrate, by difference (g)": "Carbohydrate",
        "Protein (g)": "Protein",
        "Total lipid (fat) (g)": "Fat",
    }
    
    NUTRIENTS_LIST = ["Carbohydrate, by difference (g)", "Protein (g)", "Total lipid (fat) (g)"]
    
    def __init__(self, cuisine_nm_dict):
        self.cuisine_nm = cuisine_nm_dict
        self.metrics = {
            'log_std': [],
            'log_skew': [],
            'powerlaw': [],
            'rescaled_Q': [],
        }
    
    def _style_axes(self, ax):
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        ax.grid(False)
    
    def _cuisine_legend(self):
        legend_handles = []
        for cuisine_name in self.cuisine_nm:
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color='w',
                    markerfacecolor=color,
                    markersize=10,
                    markeredgecolor='black',
                    label=cuisine_name,
                )
            )
        return legend_handles
    
    def plot_Qxn_individual(self, nutrient, nbins=15, figsize=(12, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        all_x_min, all_x_max, all_y_max = [], [], []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            hist = nm.Qxn(nutrient, nbins=nbins)
            
            ax.scatter(hist['centers'], hist['Q'], c=color, s=80, alpha=0.8, edgecolors='black', linewidths=0.5, label=cuisine_name, zorder=3)
            ax.plot(hist['x_fit'], hist['Q_fit'], color=color, lw=2, alpha=0.7, linestyle='--')
            
            all_x_min.append(hist['centers'].min())
            all_x_max.append(hist['centers'].max())
            all_y_max.append(max(hist['Q'].max(), hist['Q_fit'].max()))
        
        x_min, x_max = min(all_x_min) * 0.5, max(all_x_max) * 2
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
    
    def plot_Qxn_combined(self, nbins=15, figsize=(14, 9)):
        fig, ax = plt.subplots(figsize=figsize)
        all_x_min, all_x_max = [], []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            
            for nutrient in self.NUTRIENTS_LIST:
                hist = nm.Qxn(nutrient, nbins=nbins)
                linestyle = self.NUTRIENT_LINESTYLES[nutrient]
                ax.plot(hist['x_fit'], hist['Q_fit'], color=color, lw=2.5, alpha=0.8, linestyle=linestyle)
                all_x_min.append(hist['x_fit'].min())
                all_x_max.append(hist['x_fit'].max())
        
        x_min, x_max = min(all_x_min) * 0.5, max(all_x_max) * 2
        
        ax.set_xscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 0.35)
        ax.set_xlabel(r"$\mathbf{x_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"$\mathbf{Q(x_n)}$", fontsize=self.label_size, fontweight='bold')

        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        cuisine_legend = [
            Line2D([0], [0], color=self.CUISINE_COLORS.get(cuisine_name, '#999999'), lw=2.5, label=cuisine_name)
            for cuisine_name in self.cuisine_nm
        ]
        ax.legend(handles=cuisine_legend, loc='upper right', framealpha=0.95, fontsize=self.legend_size, ncol=2)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_log_std_combined(self, figsize=(14, 9)):
        fig, ax = plt.subplots(figsize=figsize)
        all_means, all_log_stds = [], []
        legend_entries_added = {c: False for c in self.CUISINE_COLORS}
        metrics_list = []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            stats = nm.all_stats()
            
            for _, row in stats.iterrows():
                nutrient = row['nutrient']
                marker = self.NUTRIENT_MARKERS.get(nutrient, 'o')
                
                label = cuisine_name if not legend_entries_added.get(cuisine_name, True) else None
                ax.scatter(row['mean'], row['log_std'], c=color, marker=marker, s=120, alpha=0.85, edgecolors='black', linewidths=1, zorder=3, label=label)
                legend_entries_added[cuisine_name] = True
                
                all_means.append(row['mean'])
                all_log_stds.append(row['log_std'])
                
                metrics_list.append({
                    'Cuisine': cuisine_name,
                    'Nutrient': self.NUTRIENT_SHORT.get(nutrient, nutrient),
                    'μₙ (g)': row['mean'],
                    'sₙ': row['log_std']
                })
        
        all_log_stds_arr = np.array(all_log_stds)
        global_mean = np.mean(all_log_stds_arr)
        global_std = np.std(all_log_stds_arr, ddof=1)
        
        ax.axhspan(global_mean - global_std, global_mean + global_std, color='gray', alpha=0.2)
        ax.axhline(y=global_mean, color='gray', linestyle='--', linewidth=2)
        

        ax.set_xlabel(r"Mean concentration, $\mathbf{\mu_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"Log standard deviation, $\mathbf{s_n}$", fontsize=self.label_size, fontweight='bold')

        ax.set_xscale('log')
        x_min, x_max = min(all_means), max(all_means)
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
        ax.legend(handles=self._cuisine_legend(), loc='lower right', framealpha=0.95, fontsize=self.legend_size, ncol=2)
        plt.tight_layout()
        
        return fig, ax, metrics_list, {'mean': global_mean, 'std': global_std}
    
    def plot_log_skew_combined(self, figsize=(14, 9)):
        fig, ax = plt.subplots(figsize=figsize)
        all_means, all_log_skews = [], []
        legend_entries_added = {c: False for c in self.CUISINE_COLORS}
        metrics_list = []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            stats = nm.all_stats()
            
            for _, row in stats.iterrows():
                nutrient = row['nutrient']
                marker = self.NUTRIENT_MARKERS.get(nutrient, 'o')
                
                label = cuisine_name if not legend_entries_added.get(cuisine_name, True) else None
                ax.scatter(row['mean'], row['log_skew'], c=color, marker=marker, s=120, alpha=0.85, edgecolors='black', linewidths=1, zorder=3, label=label)
                legend_entries_added[cuisine_name] = True
                
                all_means.append(row['mean'])
                all_log_skews.append(row['log_skew'])
                
                metrics_list.append({
                    'Cuisine': cuisine_name,
                    'Nutrient': self.NUTRIENT_SHORT.get(nutrient, nutrient),
                    'μₙ (g)': row['mean'],
                    'skₙ': row['log_skew']
                })
        
        ax.axhline(y=0, color='red', linestyle=':', linewidth=2, alpha=0.5, zorder=2)
        
        ax.set_xlabel(r"Mean concentration, $\mathbf{\mu_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"Log-skewness, $\mathbf{sk_n}$", fontsize=self.label_size, fontweight='bold')

        ax.set_xscale('log')
        x_min, x_max = min(all_means), max(all_means)
        ax.set_xlim(x_min * 0.3, x_max * 3)

        ax.set_ylim(-2, 2)
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        ax.legend(handles=self._cuisine_legend(), loc='upper right', framealpha=0.95, fontsize=self.legend_size, ncol=2)
        plt.tight_layout()
        
        return fig, ax, metrics_list
    
    def plot_rescaled_Q_individual(self, nutrient, nbins=15, figsize=(12, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        short_name = self.NUTRIENT_SHORT.get(nutrient, nutrient)
        metrics_list = []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            res = nm.rescaled_Q(nutrient, nbins=nbins)
            
            mu_fit, sigma_fit = norm.fit(res['log_y'])
            x_fit = np.linspace(-8, 8, 1000)
            
            ax.plot(x_fit, norm.pdf(x_fit, mu_fit, sigma_fit), '--', color=color, lw=2, alpha=0.8)
            ax.scatter(res['centers'], res['Q'], s=70, alpha=0.8, color=color, edgecolors='black', linewidths=0.5, label=cuisine_name, zorder=3)
            
            metrics_list.append({
                'Cuisine': cuisine_name,
                'Nutrient': short_name,
                'mₙ': res['m_n'],
                'sₙ': res['s_n'],
                'translated_mean': np.mean(res['log_y']),
                'translated_std': np.std(res['log_y'], ddof=1),
                'fit_σ': sigma_fit
            })
        
        ax.set_xlabel(r'$\mathbf{log(y_n)}$', fontsize=self.tick_size, fontweight='bold')
        ax.set_ylabel(r'$\mathbf{Q(y_n)}$', fontsize=self.tick_size, fontweight='bold')
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)
        ax.legend(loc='upper left', framealpha=0.95, fontsize=self.legend_size, ncol=2)
        plt.tight_layout()
        
        return fig, ax, metrics_list
    
    def plot_rescaled_Q_combined(self, nbins=15, figsize=(14, 9)):
        fig, ax = plt.subplots(figsize=figsize)
        metrics_list = []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            
            for nutrient in self.NUTRIENTS_LIST:
                res = nm.rescaled_Q(nutrient, nbins=nbins)
                short_name = self.NUTRIENT_SHORT.get(nutrient, nutrient)
                linestyle = self.NUTRIENT_LINESTYLES[nutrient]
                
                mu_fit, sigma_fit = norm.fit(res['log_y'])
                x_fit = np.linspace(-8, 8, 1000)
                
                ax.plot(x_fit, norm.pdf(x_fit, mu_fit, sigma_fit), color=color, lw=2.5, alpha=0.8, linestyle=linestyle)
                
                metrics_list.append({
                    'Cuisine': cuisine_name,
                    'Nutrient': short_name,
                    'mₙ': res['m_n'],
                    'sₙ': res['s_n'],
                    'translated_mean': np.mean(res['log_y']),
                    'translated_std': np.std(res['log_y'], ddof=1),
                    'fit_σ': sigma_fit
                })
        
        ax.set_xlabel(r'$\mathbf{log(y_n)}$', fontsize=self.tick_size, fontweight='bold')
        ax.set_ylabel(r'$\mathbf{Q(y_n)}$', fontsize=self.tick_size, fontweight='bold')
        
        ax.set_xlim(-8, 8)
        ax.set_ylim(0, 0.5)
        
        ax.tick_params(axis='both', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.grid(False)

        cuisine_legend = [
            Line2D([0], [0], color=self.CUISINE_COLORS.get(cuisine_name, '#999999'), lw=2.5, label=cuisine_name)
            for cuisine_name in self.cuisine_nm
        ]
        ax.legend(handles=cuisine_legend, loc='upper right', framealpha=0.95, fontsize=self.legend_size, ncol=2)
        plt.tight_layout()
        
        return fig, ax, metrics_list
    
    def plot_powerlaw_combined(self, figsize=(14, 10)):
        fig, ax = plt.subplots(figsize=figsize)
        all_means, all_stds = [], []
        legend_entries_added = {c: False for c in self.CUISINE_COLORS}
        metrics_list = []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
            stats = nm.all_stats()
            x_vals, y_vals = [], []
            
            for _, row in stats.iterrows():
                nutrient = row['nutrient']
                marker = self.NUTRIENT_MARKERS.get(nutrient, 'o')
                
                label = cuisine_name if not legend_entries_added.get(cuisine_name, True) else None
                ax.scatter(row['mean'], row['std'], c=color, marker=marker, s=120, alpha=0.85, edgecolors='black', linewidths=1, zorder=3, label=label)
                legend_entries_added[cuisine_name] = True
                
                x_vals.append(row['mean'])
                y_vals.append(row['std'])
                all_means.append(row['mean'])
                all_stds.append(row['std'])
            
            x_fit = np.log(np.array(x_vals))
            y_fit = np.log(np.array(y_vals))
            valid_idx = np.isfinite(x_fit) & np.isfinite(y_fit)
            
            if np.sum(valid_idx) >= 2:
                slope, intercept = np.polyfit(x_fit[valid_idx], y_fit[valid_idx], 1)
                R_squared = np.corrcoef(x_fit[valid_idx], y_fit[valid_idx])[0, 1]**2
                
                x_min_c, x_max_c = min(x_vals), max(x_vals)
                x_line = np.logspace(np.log10(x_min_c * 0.5), np.log10(x_max_c * 2), 100)
                ax.plot(x_line, np.exp(intercept) * x_line**slope, color=color, lw=2, alpha=0.7, linestyle='-')
                
                metrics_list.append({
                    'Cuisine': cuisine_name,
                    'Slope': round(slope, 4),
                    'Intercept': round(intercept, 4),
                    'R²': round(R_squared, 4)
                })
        
        x_min, x_max = min(all_means), max(all_means)
        y_min, y_max = min(all_stds), max(all_stds)
        
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(r"Mean concentration, $\mathbf{\mu_n}$ (g)", fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel(r"Standard deviation, $\mathbf{\sigma_n}$", fontsize=self.label_size, fontweight='bold')

        ax.set_xlim(x_min * 0.3, x_max * 5)
        ax.set_ylim(y_min * 0.3, y_max * 5)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='both', which='major', labelsize=self.tick_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')

        cuisine_legend = [
            Line2D([0], [0], color=self.CUISINE_COLORS.get(cuisine_name, '#999999'), lw=2.5, label=cuisine_name)
            for cuisine_name in self.cuisine_nm
        ]
        ax.legend(handles=cuisine_legend, loc='upper left', framealpha=0.95, fontsize=self.legend_size, ncol=2)
        
        ax.grid(False)
        plt.tight_layout()
        
        return fig, ax, metrics_list
    
    def _compute_spread_diff(self, nm, distribution, n_samples=10000):
        sample_funcs = {
            'lognormal': self._sample_lognormal,
            'weibull': self._sample_weibull,
            'gamma': self._sample_gamma,
            'gaussian': self._sample_gaussian,
            'exponential': self._sample_exponential,
            'uniform': self._sample_uniform,
        }
        
        if distribution not in sample_funcs:
            return np.inf
        
        stats_df = nm.all_stats()
        cv_real = stats_df['std'].values / stats_df['mean'].values
        
        mu_sim, sigma_sim = [], []
        func = sample_funcs[distribution]
        for nutrient in nm.nutrients:
            try:
                mu, sigma = func(nm, nutrient, n_samples)
                mu_sim.append(mu)
                sigma_sim.append(sigma)
            except:
                mu_sim.append(np.nan)
                sigma_sim.append(np.nan)
        
        cv_sim = np.array(sigma_sim) / np.array(mu_sim)
        valid_cv = np.isfinite(cv_real) & np.isfinite(cv_sim) & (cv_real > 0) & (cv_sim > 0)
        
        if np.sum(valid_cv) < 2:
            return np.inf
        
        spread_diff = abs(np.std(np.log(cv_real[valid_cv])) - np.std(np.log(cv_sim[valid_cv])))
        return spread_diff
    
    def _sample_lognormal(self, nm, nutrient, n_samples=10000):
        shape, loc, scale = nm.ks_test(nutrient)['lognormal']['params']
        samples = lognorm.rvs(shape, loc=loc, scale=scale, size=n_samples)
        samples = samples[samples > 0]
        return samples.mean(), samples.std()
    
    def _sample_weibull(self, nm, nutrient, n_samples=10000):
        c, loc, scale = nm.ks_test(nutrient)['weibull']['params']
        samples = weibull_min.rvs(c, loc=loc, scale=scale, size=n_samples)
        samples = samples[samples > 0]
        return samples.mean(), samples.std()
    
    def _sample_gamma(self, nm, nutrient, n_samples=10000):
        a, loc, scale = nm.ks_test(nutrient)['gamma']['params']
        samples = gamma.rvs(a, loc=loc, scale=scale, size=n_samples)
        samples = samples[samples > 0]
        return samples.mean(), samples.std()
    
    def _sample_gaussian(self, nm, nutrient, n_samples=10000):
        mu_fit, sigma_fit = nm.ks_test(nutrient)['gaussian']['params']
        samples = np.abs(norm.rvs(loc=mu_fit, scale=sigma_fit, size=n_samples))
        return samples.mean(), samples.std()
    
    def _sample_exponential(self, nm, nutrient, n_samples=10000):
        loc_e, scale_e = nm.ks_test(nutrient)['exponential']['params']
        samples = expon.rvs(loc=loc_e, scale=scale_e, size=n_samples)
        return samples.mean(), samples.std()
    
    def _sample_uniform(self, nm, nutrient, n_samples=10000):
        loc_u, scale_u = nm.ks_test(nutrient)['uniform']['params']
        samples = uniform.rvs(loc=loc_u, scale=scale_u, size=n_samples)
        return samples.mean(), samples.std()
    
    def plot_ks_distance_cdf_combined(self, figsize=(12, 8)):
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        
        distributions = ['lognormal', 'weibull', 'gamma', 'gaussian', 'exponential', 'uniform']
        dist_col_map = {
            'lognormal': 'D_lognormal',
            'weibull': 'D_weibull',
            'gamma': 'D_gamma',
            'gaussian': 'D_gaussian',
            'exponential': 'D_exponential',
            'uniform': 'D_uniform'
        }
        
        all_d_min = 1.0
        best_dist_summary = []
        
        for cuisine_name, nm in self.cuisine_nm.items():
            ks_df = nm.all_ks_tests()
            
            mean_d_stats = {}
            for dist in distributions:
                d_col = dist_col_map[dist]
                if d_col in ks_df.columns:
                    mean_d_stats[dist] = ks_df[d_col].mean()
                else:
                    mean_d_stats[dist] = np.inf
            
            best_dist = min(mean_d_stats, key=mean_d_stats.get)
            best_mean_d = mean_d_stats[best_dist]
            
            best_dist_summary.append({
                'Cuisine': cuisine_name,
                'Best Distribution': best_dist.capitalize(),
                'Mean KS D': round(best_mean_d, 4)
            })
            
            d_col = dist_col_map[best_dist]
            
            if d_col in ks_df.columns:
                d_values = ks_df[d_col].dropna().values
                if len(d_values) >= 2:
                    all_d_min = min(all_d_min, d_values.min())
                    
                    kde = gaussian_kde(d_values, bw_method='scott')
                    x_smooth = np.linspace(max(0, d_values.min() * 0.9), 1, 200)
                    cdf_smooth = np.array([kde.integrate_box_1d(max(0, d_values.min() * 0.9), x) for x in x_smooth])
                    cdf_smooth = cdf_smooth / cdf_smooth[-1] if cdf_smooth[-1] > 0 else cdf_smooth
                    
                    color = self.CUISINE_COLORS.get(cuisine_name, '#999999')
                    ax.plot(x_smooth, cdf_smooth, color=color, lw=2.5, linestyle='-', 
                            label=f"{cuisine_name} ({best_dist.capitalize()})", alpha=0.85, zorder=2)
        
        ax.axvline(x=0.1, color='#333333', linestyle='--', lw=2.5, alpha=0.9, zorder=1)
        ax.axvline(x=0.2, color='#333333', linestyle=':', lw=2.5, alpha=0.9, zorder=1)
        ax.text(0.1, 1.02, 'D=0.1', ha='center', va='bottom', fontsize=self.label_size, fontweight='bold', color='#333333')
        ax.text(0.2, 1.02, 'D=0.2', ha='center', va='bottom', fontsize=self.label_size, fontweight='bold', color='#333333')
        
        ax.set_xlabel('Kolmogorov-Smirnov Distance', fontsize=self.label_size, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=self.label_size, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.08)
        
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        ax.tick_params(axis='both', which='major', labelsize=13)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        ax.legend(loc='lower right', fontsize=self.legend_size, framealpha=0.95, edgecolor='#cccccc', fancybox=True, ncol=1)
        ax.grid(False)
        plt.tight_layout()
        
        print("\nBest Distribution for Each Cuisine:")
        summary_df = pd.DataFrame(best_dist_summary)
        print(summary_df.to_string(index=False))
        
        return fig, ax, summary_df
    
    def save_metrics_to_csv(self, log_std_metrics, log_skew_metrics, powerlaw_metrics, rescaled_metrics, output_dir='DATA/PROCESSED'):
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        if log_std_metrics:
            df = pd.DataFrame(log_std_metrics)
            path = f'{output_dir}/Combined_Log_Std_Metrics.csv'
            df.to_csv(path, index=False, float_format='%.4f')
            results['log_std'] = path
        
        if log_skew_metrics:
            df = pd.DataFrame(log_skew_metrics)
            path = f'{output_dir}/Combined_Log_Skewness_Metrics.csv'
            df.to_csv(path, index=False, float_format='%.4f')
            results['log_skew'] = path
        
        if powerlaw_metrics:
            df = pd.DataFrame(powerlaw_metrics)
            path = f'{output_dir}/Combined_PowerLaw_Metrics.csv'
            df.to_csv(path, index=False, float_format='%.4f')
            results['powerlaw'] = path
        
        if rescaled_metrics:
            df = pd.DataFrame(rescaled_metrics)
            path = f'{output_dir}/Combined_Rescaled_Q_Metrics.csv'
            df.to_csv(path, index=False, float_format='%.4f')
            results['rescaled_Q'] = path
        
        return results
