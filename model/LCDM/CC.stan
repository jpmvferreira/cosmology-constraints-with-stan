functions {
}

data {
  int<lower=0> N1;
  array[N1] real z;
  array[N1] real Hobs;   // provided in km s⁻¹ Mpc⁻¹
  array[N1] real error;  // .
}

transformed data {
}

parameters {
  real<lower=0> h;  // defined as H₀ = 100h km s⁻¹ Mpc⁻¹
  real<lower=0> Om;
}

transformed parameters {
  array[N1] real H;
  for (i in 1:N1) {
      H[i] = 100*h*sqrt(Om*(1+z[i])^3 + (1-Om));
  }
}

model {
  // priors
  h ~ normal(0.7, 10);
  Om ~ normal(0.3, 10);

  // likelihood
  H ~ normal(Hobs, error);
}
