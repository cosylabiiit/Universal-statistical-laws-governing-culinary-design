import numpy as np
import pandas as pd
from scipy.stats import norm, skew, lognorm, gamma, weibull_min, kstest, expon, uniform


class NutrientMatrix:
    def __init__(self, df):
        self.df = df
        self.n_foods = df.shape[0]
        self.n_nutrients = df.shape[1]
        self.nutrients = df.columns.tolist()
        self._cache = {}
        self._stats_df = None
    
    @property
    def prevalence(self):
        # Fraction of foods with non-zero value for each nutrient
        return (self.df > 0).sum(axis=0) / self.n_foods
    
    def prevalence_table(self, ascending=False):
        # Returns DataFrame of nutrients sorted by prevalence
        prev = self.prevalence
        table = pd.DataFrame({"Nutrient": prev.index, "p_n": prev.values})
        return table.sort_values("p_n", ascending=ascending).reset_index(drop=True)
    
    def positive(self, nutrient):
        # Returns only positive values for a nutrient
        v = self.df[nutrient].values
        return v[v > 0]
    
    def log_values(self, nutrient):
        # Log-transformed positive values
        return np.log(self.positive(nutrient))
    
    def stats(self, nutrient):
        # Compute mean, std, log-stats for a single nutrient (cached)
        if nutrient in self._cache:
            return self._cache[nutrient]
        
        v = self.positive(nutrient)
        logv = np.log(v)
        
        result = {
            'mean': np.mean(v),
            'std': np.std(v, ddof=1),
            'log_mean': np.mean(logv),
            'log_std': np.std(logv, ddof=1),
            'log_skew': skew(logv, bias=False),
            'count': len(v),
        }
        self._cache[nutrient] = result
        return result
    
    def all_stats(self):
        # DataFrame of stats for all nutrients
        if self._stats_df is not None:
            return self._stats_df
        
        records = [{'nutrient': n, **self.stats(n)} for n in self.nutrients]
        self._stats_df = pd.DataFrame(records)
        return self._stats_df
    
    def mean_log_std(self):
        # Average log-std across all nutrients
        return self.all_stats()['log_std'].mean()
    
    def log_std_summary(self):
        # Summary statistics for log-std distribution
        s = self.all_stats()['log_std']
        return {'mean': s.mean(), 'std': s.std(), 'min': s.min(), 'max': s.max(), 'median': s.median()}
    
    def std_summary(self):
        # Summary statistics for raw std distribution
        s = self.all_stats()['std']
        return {'mean': s.mean(), 'std': s.std(), 'min': s.min(), 'max': s.max(), 'median': s.median()}
    
    def log_skew_summary(self):
        # Summary statistics for log-skewness distribution
        s = self.all_stats()['log_skew']
        return {'mean': s.mean(), 'std': s.std(), 'min': s.min(), 'max': s.max(), 'median': s.median()}
    
    def Qxn(self, nutrient, nbins=15):
        # Histogram Q(x_n) in log-space with Gaussian fit
        logv = self.log_values(nutrient)
        
        Q, edges = np.histogram(logv, bins=nbins, density=True)
        centers = np.exp((edges[:-1] + edges[1:]) / 2)
        
        mu, sigma = norm.fit(logv)
        xfit_log = np.linspace(logv.min(), logv.max(), 1000)
        
        return {
            'centers': centers, 'Q': Q, 'edges': edges,
            'x_fit': np.exp(xfit_log), 'Q_fit': norm.pdf(xfit_log, mu, sigma),
            'mu': mu, 'sigma': sigma,
        }
    
    def rescaled_Q(self, nutrient, nbins=15):
        # Rescaled distribution: y_n = exp(log(x_n) - m_n)
        logv = self.log_values(nutrient)
        m_n = np.mean(logv)
        s_n = np.std(logv, ddof=1)
        log_y = logv - m_n
        
        Q, edges = np.histogram(log_y, bins=nbins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        
        return {'centers': centers, 'Q': Q, 'edges': edges, 'm_n': m_n, 's_n': s_n, 'log_y': log_y}
    
    def ks_test(self, nutrient):
        # K-S test against multiple distributions
        v = self.positive(nutrient)
        results = {}
        
        # Lognormal fit
        shape, loc, scale = lognorm.fit(v, floc=0)
        D, p = kstest(v, 'lognorm', args=(shape, loc, scale))
        results['lognormal'] = {'D': D, 'p': p, 'params': (shape, loc, scale)}
        
        # Gamma fit
        a, loc, scale = gamma.fit(v, floc=0)
        D, p = kstest(v, 'gamma', args=(a, loc, scale))
        results['gamma'] = {'D': D, 'p': p, 'params': (a, loc, scale)}
        
        # Weibull fit
        c, loc, scale = weibull_min.fit(v, floc=0)
        D, p = kstest(v, 'weibull_min', args=(c, loc, scale))
        results['weibull'] = {'D': D, 'p': p, 'params': (c, loc, scale)}
        
        # Gaussian fit
        mu, sigma = norm.fit(v)
        D, p = kstest(v, 'norm', args=(mu, sigma))
        results['gaussian'] = {'D': D, 'p': p, 'params': (mu, sigma)}

        # Exponential fit
        loc_e, scale_e = expon.fit(v, floc=0)
        D, p = kstest(v, 'expon', args=(loc_e, scale_e))
        results['exponential'] = {'D': D, 'p': p, 'params': (loc_e, scale_e)}

        # Uniform fit
        loc_u, scale_u = uniform.fit(v)
        D, p = kstest(v, 'uniform', args=(loc_u, scale_u))
        results['uniform'] = {'D': D, 'p': p, 'params': (loc_u, scale_u)}
        
        return results
    
    def all_ks_tests(self):
        # K-S test results for all nutrients as DataFrame
        records = []
        for n in self.nutrients:
            ks = self.ks_test(n)
            records.append({
                'nutrient': n, 'n_samples': len(self.positive(n)),
                'D_lognormal': ks['lognormal']['D'], 'P_lognormal': ks['lognormal']['p'],
                'D_gamma': ks['gamma']['D'], 'P_gamma': ks['gamma']['p'],
                'D_weibull': ks['weibull']['D'], 'P_weibull': ks['weibull']['p'],
                'D_gaussian': ks['gaussian']['D'], 'P_gaussian': ks['gaussian']['p'],
                'D_exponential': ks['exponential']['D'], 'P_exponential': ks['exponential']['p'],
                'D_uniform': ks['uniform']['D'], 'P_uniform': ks['uniform']['p'],
            })
        return pd.DataFrame(records)
    
    def linear_fit(self):
        # Linear fit of log(std) vs log(mean) for Taylor's law
        df = self.all_stats()
        valid = (df['mean'] > 0) & (df['std'] > 0)
        x, y = np.log(df.loc[valid, 'mean']), np.log(df.loc[valid, 'std'])
        
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        
        slope, intercept = np.polyfit(x, y, 1)
        return {'slope': slope, 'intercept': intercept, 'R_squared': np.corrcoef(x, y)[0, 1] ** 2}
    
    def __repr__(self):
        return f"NutrientMatrix(n_foods={self.n_foods}, n_nutrients={self.n_nutrients})"
