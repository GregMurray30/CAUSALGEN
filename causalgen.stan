functions {

  real softplus(real x) {

      return log1p_exp(x);

    }

}

 

data {

  int<lower=1> N_pre;

  int<lower=1> N_post;

  int<lower=1> num_events;

 

  matrix[N_pre+N_post, num_events] Evts;

 

  vector[N_pre] Y_csl;

  vector[N_pre] Y_hc;

 

  // Priors

  real mu0_trend_mn;

  real mu0_trend_sd;

  real beta0_trend_mn;

  real beta0_trend_sd;

  real sigma_level_trend_mn;

  real sigma_level_trend_sd;

  real sigma_slope_trend_mn;

  real sigma_slope_trend_sd;

  real epsilon_trend_sd;

  real eta_trend_sd;

 

  real rho1_mn;

  real rho1_sd;

  real rho2_mn;

  real rho2_sd;

  real rho3_mn;

  real rho3_sd;

//real theta_mn;

  //real theta_sd;

 

  real sigma_shared_sd;

}

 
transformed data {

  int<lower=1> T = N_pre + N_post;

}

 

parameters {

  //real<lower=0.01> scale_hc;

  real mu0_trend_shared;

  real beta0_trend_shared;

  real sigma_level_trend_shared;

  real sigma_slope_trend_shared;

 

  // Segment-specific trend levels and slopes

  real mu0_trend_csl;

  real mu0_trend_hc;

  real beta0_trend_csl;

  real beta0_trend_hc;

 

  // Shared noise scale priors (if pooling epsilon/eta across segments)

  real<lower=0> sigma_level_trend_csl;

  real<lower=0> sigma_level_trend_hc;

 

  real<lower=0> sigma_slope_trend_csl;

  real<lower=0> sigma_slope_trend_hc;

 

  // Innovation noise

  vector[N_pre] epsilon_trend_csl;

  vector[N_pre] epsilon_trend_hc;

  vector[N_pre] eta_trend_csl;

  vector[N_pre] eta_trend_hc;




  vector[T] delta_trend_csl_raw;

  vector[T] delta_trend_hc_raw;

  real sigma_dev_trend;

 

  // Shared ARMA latent state

  real<lower=0> sigma_shared;

  vector[N_pre] shared_raw;

  real<lower=-1, upper=.8> rho1;

  real<lower=-1, upper=.1> rho2;

  real<lower=-1, upper=.1> rho3;

//  real<lower=-1, upper=1> theta;

 

  // Geo hierarchy

  //vector[geo_n] geo_state_offset_raw;

//real<lower=0> sigma_geo;

 

  // Segment-level deviations

  vector[N_pre] delta_csl_raw;

  vector[N_pre] delta_hc_raw;

  real<lower=0> sigma_dev_csl;

  real<lower=0> sigma_dev_hc;

  real<lower=-1, upper=1> rho_seg;

 

  // Covariates

  vector<lower=0>[num_events] beta_event;

 

  // Observation noise

  real<lower=0> sigma_obs_csl;

  real<lower=0> sigma_obs_hc;

}

 

transformed parameters {

  vector[T] level_trend_raw;

  vector[T] slope_trend;

  vector[T] trend_csl;

  vector[T] trend_hc;

 

  vector[T] state_shared;

  vector[T] geo_state_adjusted;

 

  vector[T] delta_csl;

  vector[T] delta_hc;

 

  vector[T] state_csl;

  vector[T] state_hc;

 

  vector[T] mu_csl;

  vector[T] mu_hc;




  // Local linear trend with partial pooling for CSL and HC // Shared trend slope with segment-specific level offsets

 

  vector[T] level_trend_csl;

  vector[T] slope_trend_csl;

  vector[T] level_trend_hc;

  vector[T] slope_trend_hc;

 

  // Initialization

  level_trend_csl[1] = mu0_trend_csl;

  slope_trend_csl[1] = beta0_trend_csl;

  level_trend_hc[1] = mu0_trend_hc;

  slope_trend_hc[1] = beta0_trend_hc;

 

  // Pre-period evolution

  for (t in 2:N_pre) {

    slope_trend_csl[t] = slope_trend_csl[t - 1] + sigma_slope_trend_csl * eta_trend_csl[t];

    level_trend_csl[t] = level_trend_csl[t - 1] + slope_trend_csl[t - 1] + sigma_level_trend_csl * epsilon_trend_csl[t];

 

    slope_trend_hc[t] = slope_trend_hc[t - 1] + sigma_slope_trend_hc * eta_trend_hc[t];

    level_trend_hc[t] = level_trend_hc[t - 1] + slope_trend_hc[t - 1] + sigma_level_trend_hc * epsilon_trend_hc[t];

  }

 

  // Post-period forward fill (no new noise)

  for (t in (N_pre + 1):T) {

    slope_trend_csl[t] = slope_trend_csl[t - 1];

    level_trend_csl[t] = level_trend_csl[t - 1] + slope_trend_csl[t - 1];

 

    slope_trend_hc[t] = slope_trend_hc[t - 1];

    level_trend_hc[t] = level_trend_hc[t - 1] + slope_trend_hc[t - 1];

  }




  // Shared state (centered at 1)

  state_shared[1] = 1 + sigma_shared * shared_raw[1];

  state_shared[2] = 1 + rho1 * (state_shared[1] - 1) + sigma_shared * shared_raw[2];

  state_shared[3] = 1 + rho1 * (state_shared[2] - 1) + rho2 * (state_shared[1] - 1) + sigma_shared * shared_raw[3];

  for (t in 4:N_pre) {

    state_shared[t] = 1

      + rho1 * (state_shared[t - 1] - 1)

      + rho2 * (state_shared[t - 2] - 1)

      + rho3 * (state_shared[t - 3] - 1)

      + sigma_shared * shared_raw[t];

  }

  for (t in (N_pre + 1):T) {

    state_shared[t] = 1

      + rho1 * (state_shared[t - 1] - 1)

      + rho2 * (state_shared[t - 2] - 1)

      + rho3 * (state_shared[t - 3] - 1);

  }

 

  geo_state_adjusted = state_shared;

 

  // Segment-level deviation (centered AR(1))

  delta_csl[1] = 1 + sigma_dev_csl * delta_csl_raw[1];

  delta_hc[1] = 1 + sigma_dev_hc * delta_hc_raw[1];

  for (t in 2:N_pre) {

    delta_csl[t] = 1 + rho_seg * (delta_csl[t - 1] - 1) + sigma_dev_csl * delta_csl_raw[t];

    delta_hc[t] = 1 + rho_seg * (delta_hc[t - 1] - 1) + sigma_dev_hc * delta_hc_raw[t];

  }

  for (t in (N_pre + 1):T) {

    delta_csl[t] = delta_csl[N_pre];

    delta_hc[t] = delta_hc[N_pre];

  }

 

  // Latent state per segment

  for (t in 1:T) {

    state_csl[t] = geo_state_adjusted[t] * delta_csl[t];

    state_hc[t] = geo_state_adjusted[t] * delta_hc[t];

 
    trend_csl[t] = level_trend_csl[t];

    trend_hc[t] = level_trend_hc[t];


    mu_csl[t] = state_csl[t] * (1 + dot_product(beta_event, Evts[t,])) * trend_csl[t];

    mu_hc[t] =  state_hc[t] * (1 + dot_product(beta_event, Evts[t,])) * trend_hc[t];

  }

}




model {

  //scale_hc ~ normal(.5, .1);

 

  // Shared priors

  mu0_trend_shared ~ normal(mu0_trend_mn, mu0_trend_sd);

  beta0_trend_shared ~ normal(beta0_trend_mn, beta0_trend_sd);

  sigma_level_trend_shared ~ normal(0, sigma_level_trend_sd);

  sigma_slope_trend_shared ~ normal(0, sigma_slope_trend_sd);

 

  // Hierarchical priors for segment-specific trend components

  mu0_trend_csl ~ normal(mu0_trend_shared, 5);

  mu0_trend_hc ~ normal(mu0_trend_shared, 5);

  beta0_trend_csl ~ normal(beta0_trend_shared, 0.3);

  beta0_trend_hc ~ normal(beta0_trend_shared, 0.3);

  sigma_level_trend_csl ~ normal(sigma_level_trend_shared, 0.4);

  sigma_level_trend_hc ~ normal(sigma_level_trend_shared, 0.6);

  sigma_slope_trend_csl ~ normal(sigma_slope_trend_shared, 0.4);

  sigma_slope_trend_hc ~ normal(sigma_slope_trend_shared, 0.6);

 

  // Innovations

  epsilon_trend_csl ~ normal(0, epsilon_trend_sd);

  epsilon_trend_hc ~ normal(0, epsilon_trend_sd);

  eta_trend_csl ~ normal(0, eta_trend_sd);

  eta_trend_hc ~ normal(0, eta_trend_sd);

 

  rho1 ~ normal(rho1_mn, rho1_sd);

  rho2 ~ normal(rho2_mn, rho2_sd);

  rho3 ~ normal(rho3_mn, rho3_sd);

  //theta ~ normal(theta_mn, theta_sd);

  sigma_shared ~ normal(0, sigma_shared_sd);

  shared_raw ~ normal(0, 1);

 

// geo_state_offset_raw ~ normal(0, 1);

  //sigma_geo ~ normal(0, 1);

 

  delta_csl_raw ~ normal(0, 1);

  delta_hc_raw ~ normal(0, 1);

  sigma_dev_csl ~ normal(0, .005);

  sigma_dev_hc ~ normal(0, .1);

  rho_seg ~ normal(0, .3);

 

  beta_event ~ normal(.01, 1);

 

  sigma_obs_csl ~ normal(2, 3);

  sigma_obs_hc ~ normal(.2, 1);




  sigma_dev_trend ~ normal(0, .2);

  delta_trend_csl_raw ~ normal(0, 1);

  delta_trend_hc_raw ~ normal(0, 1);

 

  // Likelihood

  for (t in 1:N_pre) {

    Y_csl[t] ~ normal(mu_csl[t], sigma_obs_csl);

    Y_hc[t] ~ normal(mu_hc[t], sigma_obs_hc);

 

  }

}

 

generated quantities {

  vector[N_pre] Y_csl_pred_pre;

  vector[N_pre] Y_hc_pred_pre;

 

  for (t in 1:N_pre) {

    Y_csl_pred_pre[t] = normal_rng(mu_csl[t], sigma_obs_csl);

    Y_hc_pred_pre[t] = normal_rng(mu_hc[t], sigma_obs_hc);

  }

 

  vector[N_post] Y_csl_pred_post;

  vector[N_post] Y_hc_pred_post;

 

  for (t in 1:N_post) {

    int idx = N_pre + t;

    Y_csl_pred_post[t] = normal_rng(mu_csl[idx], sigma_obs_csl);

    Y_hc_pred_post[t] = normal_rng(mu_hc[idx], sigma_obs_hc);

  }

}

