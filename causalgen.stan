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




  // Geo hierarchy

  //vector[geo_n] geo_state_offset_raw;

//real<lower=0> sigma_geo;

 

real<lower=-1, upper=1> rho1_shared;

 real<lower=-1, upper=1> rho2_shared;

 real<lower=-1, upper=1> rho3_shared;

 //real<lower=-1, upper=1> theta_shared;

 

  real<lower=0> sigma_rho;

 

  real<lower=-1, upper=1> rho1_csl;

  real<lower=-1, upper=1> rho2_csl;

  real<lower=-1, upper=1> rho3_csl;

  //real theta_csl;

 

  real<lower=-1, upper=1> rho1_hc;

  real<lower=-1, upper=1> rho2_hc;

  real<lower=-1, upper=1> rho3_hc;

  //real theta_hc;

 

  vector[N_pre] state_csl_raw;

  vector[N_pre] state_hc_raw;

 

  real<lower=0> sigma_state_shared;

  real<lower=0> sigma_state_csl;

  real<lower=0> sigma_state_hc;

 

  //real sigma_delta_csl;

  //real sigma_delta_hc;

 

  vector[N_pre] state_shared_raw;

 

  //vector[T] delta_csl_raw;

// vector[T] delta_hc_raw;

 

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

 

  vector[T] geo_state_adjusted;

 

  //vector[T] delta_csl;

  //vector[T] delta_hc;

 

  vector[T] state_csl;

  vector[T] state_hc;

 

  vector[T] mu_csl;

  vector[T] mu_hc;

 

  vector[T] delta_mean;

 

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



  // Initialize states

  state_csl[1] = 1 + sigma_state_csl * state_shared_raw[1];

  state_csl[2] = 1 + rho1_csl * (state_csl[1] - 1) + sigma_state_csl * state_shared_raw[2];

  state_csl[3] = 1 + rho1_csl * (state_csl[2] - 1) + rho2_csl * (state_csl[1] - 1) + sigma_state_csl * state_shared_raw[3];

 

  state_hc[1] = 1 + sigma_state_hc * state_shared_raw[1];

  state_hc[2] = 1 + rho1_hc * (state_hc[1] - 1) + sigma_state_hc * state_shared_raw[2];

  state_hc[3] = 1 + rho1_hc * (state_hc[2] - 1) + rho2_hc * (state_hc[1] - 1) + sigma_state_hc * state_shared_raw[3];

 

  for (t in 4:N_pre) {

    state_csl[t] = 1 + rho1_csl * (state_csl[t - 1] - 1) + rho2_csl * (state_csl[t - 2] - 1) + rho3_csl * (state_csl[t - 3] - 1) + sigma_state_csl * state_shared_raw[t];

    state_hc[t] = 1 + rho1_hc * (state_hc[t - 1] - 1) + rho2_hc * (state_hc[t - 2] - 1) + rho3_hc * (state_hc[t - 3] - 1) + sigma_state_hc * state_shared_raw[t];

  }

 

  for (t in (N_pre + 1):T) {

    state_csl[t] = 1 + rho1_csl * (state_csl[t - 1] - 1) + rho2_csl * (state_csl[t - 2] - 1) + rho3_csl * (state_csl[t - 3] - 1);

    state_hc[t] = 1 + rho1_hc * (state_hc[t - 1] - 1) + rho2_hc * (state_hc[t - 2] - 1) + rho3_hc * (state_hc[t - 3] - 1);

  }

 

 

 

  state_csl = state_csl / mean(state_csl[1:N_pre]);

  state_hc = state_hc / mean(state_hc[1:N_pre]);




  // Latent state per segment

  for (t in 1:T) {

   

    trend_csl[t] = level_trend_csl[t];

    trend_hc[t] = level_trend_hc[t];




    mu_csl[t] = state_csl[t] * (1 + dot_product(beta_event, Evts[t,])) * trend_csl[t];

    mu_hc[t] =  state_hc[t] * (1 + dot_product(beta_event, Evts[t,])) * trend_hc[t];

  }

 

}




model {

 

  // Shared priors

  mu0_trend_shared ~ normal(mu0_trend_mn, mu0_trend_sd);

  beta0_trend_shared ~ normal(beta0_trend_mn, beta0_trend_sd);

  sigma_level_trend_shared ~ normal(0, sigma_level_trend_sd);

  sigma_slope_trend_shared ~ normal(0, sigma_slope_trend_sd);

 

  // Hierarchical priors for segment-specific trend components

  mu0_trend_csl ~ normal(mu0_trend_shared, 5);

  mu0_trend_hc ~ normal(mu0_trend_shared, 5);

  beta0_trend_csl ~ normal(beta0_trend_shared, 0.5);

  beta0_trend_hc ~ normal(beta0_trend_shared, 1);

  sigma_level_trend_csl ~ normal(sigma_level_trend_shared, 0.4);

  sigma_level_trend_hc ~ normal(sigma_level_trend_shared, 0.6);

  sigma_slope_trend_csl ~ normal(sigma_slope_trend_shared, 0.4);

  sigma_slope_trend_hc ~ normal(sigma_slope_trend_shared, 0.6);

 

  // Innovations

  epsilon_trend_csl ~ normal(0, epsilon_trend_sd);

  epsilon_trend_hc ~ normal(0, epsilon_trend_sd);

  eta_trend_csl ~ normal(0, eta_trend_sd);

  eta_trend_hc ~ normal(0, eta_trend_sd);

 

  state_shared_raw ~ normal(0, 1);

  sigma_state_shared ~ normal(0.01, sigma_shared_sd);

  sigma_state_csl ~ normal(sigma_state_shared, .1);

  sigma_state_hc ~ normal(sigma_state_shared, .3);

 

// geo_state_offset_raw ~ normal(0, 1);

  //sigma_geo ~ normal(0, 1);

 

  sigma_rho ~ normal(.1, .1);

 

  // AR parameter priors

  rho1_csl ~ normal(rho1_shared, sigma_rho);

  rho2_csl ~ normal(rho2_shared, sigma_rho);

  rho3_csl ~ normal(rho3_shared, sigma_rho);

  //theta_csl ~ normal(theta_shared, sigma_rho);

 

  rho1_hc ~ normal(rho1_shared, sigma_rho);

  rho2_hc ~ normal(rho2_shared, sigma_rho);

  rho3_hc ~ normal(rho3_shared, sigma_rho);

  //theta_hc ~ normal(theta_shared, sigma_rho);




  beta_event ~ normal(.1, 1);

 

  sigma_obs_csl ~ normal(2, 1);

  sigma_obs_hc ~ normal(2, 1);




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

