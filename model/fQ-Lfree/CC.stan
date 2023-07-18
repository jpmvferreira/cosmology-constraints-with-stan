// implementation of the model f(Q) = Qe^(λQ₀/Q) to be constrained using CC
// neglects the effects of radiation
// due to the usage of the DAE solver, redshifts must be provided by asceding order
// for more information, see arXiv:2306.10176

functions {
  // define the residual to be minimized by the algebraic equation solver
  // doc: https://mc-stan.org/docs/functions-reference/functions-algebraic-solver.html
  vector residual(vector state, real z, array[] real theta) {
    // fetch relevant parameters and derived parameters
    real Om = theta[2];
    real lambda = theta[3];

    real E = state[1];
    real lhs = (E^2 - 2*lambda)*exp(lambda/E^2);  // lfs of first Friedmann equation
    real rhs = Om*(1+z)^3;                        // rhs of first Friedmann equation

    // compute the residuals
    vector[1] res;
    res[1] = rhs - lhs;       // E satisfies this algebraic relation, meaning that res[1] should be 0

    return res;
  }
}

data {
  int<lower=0> N1;
  array[N1] real z;
  array[N1] real Hobs;   // provided in km s⁻¹ Mpc⁻¹
  array[N1] real error;  // .
}

parameters {
  real<lower=0> h;  // defined as H₀ = 100h km s⁻¹ Mpc⁻¹
  real<lower=0> Om;
}

transformed parameters {
  // compute λ
  real lambda;
  lambda = 0.5 + lambert_w0( -Om/(2*exp(0.5)) );

  // pseudo-parameter array
  array[3] real theta = {h, Om, lambda};

  // compute H(z)
  real E;
  vector[1] guess;
  array[N1] real H;
  for (i in 1:N1) {
    // solve the algebraic equation to obtain E(z)
    // initial guess will be the value of E(z) given by ΛCDM
    guess[1] = sqrt(Om*(1+z[i])^3 + (1-Om));
    E = solve_newton(residual, guess, z[i], theta)[1];

    H[i] = 100*h*E;
  }
}

model {
  // priors
  h ~ normal(0.7, 3);
  Om ~ normal(0.3, 3);

  // likelihood
  H ~ normal(Hobs, error);
}
