// block for user defined functions
functions {
  real integrand(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i) {
    real h = theta[1];
    real Om = theta[2];

    return 1/(Om*(1+x)^3 + 1 - Om)^0.5;
  }
}

// block to declare the variables that will hold the data being used
data {
  int<lower=0> N1;
  array[N1] real redshift;
  array[N1] real dLobs;
  array[N1] real error;
}

// process the data declared in the previous block and/or defined new variables that are related to data
// will run once at the beginning of each chain
transformed data {
  array[0] real x_r;  // dummy variables required because of integrate_1d signature
  array[0] int x_i;   // see: https://mc-stan.org/docs/functions-reference/functions-1d-integrator.html
}

parameters {
  real<lower=0> h;
  real<lower=0> Om;
}

// allows new variables to be defined in terms of data and/or parameters, this is where you should compute your model's predictions
// will be evaluated on each step
transformed parameters {
  array[2] real theta = {h, Om};

  array[N1] real dL;
  for (i in 1:N1) {
    // using c/H₀ ≈ 2.9979/h (Gpc)
    dL[i] = (1+redshift[i]) * (2.9979/h) * integrate_1d(integrand, 0, redshift[i], theta, x_r, x_i);
  }
}

// likelihood and priors
// will be evaluated on each step
model {
  // priors
  h ~ normal(0.7, 3);
  Om ~ normal(0.3, 3);

  // likelihood
  dL ~ normal(dLobs, error);
}
