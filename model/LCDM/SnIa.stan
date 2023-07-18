functions {
  // define 1/E(z)
  real Einverse(real z, real xc, array[] real theta, array[] real x_r, array[] int x_i) {
    real Om = theta[1];
    return (Om*(1+z)^3 + (1-Om))^(-0.5);
  }
}

data {
  // redshift, magnitude and corresponding error for each supernova event
  int<lower=0> N1;
  array[N1] real zcmb;
  array[N1] real mb;
  array[N1] real dmb;
}

transformed data {
  // null arrays that are required for the integrate_1d function signature
  array[0] real x_r;
  array[0] int x_i;
}

parameters {
  // define the parameters, in this case its just the matter density because 'h' is taken out
  real<lower=0,upper=1> Om;
}

transformed parameters {
  // Delta will be the difference between the supernova magnitude and the luminosity distance without the c/H₀
  // that distance will be given by the varible 'dlhat'
  real Delta;
  real dLhat;

  // compute A, B and C
  real A = 0;
  real B = 0;
  real C = 0;
  for (i in 1:N1) {
    dLhat = (1+zcmb[i]) * integrate_1d(Einverse, 0, zcmb[i], {Om}, x_r, x_i);
    Delta = mb[i] - 5*log10(dLhat);

    A += Delta^2/dmb[i]^2;
    B += Delta/dmb[i]^2;
    C += 1/dmb[i]^2;
  }
}

model {
  // priors
  Om ~ normal(0.3, 3);

  // target is the variable that holds to logarithm of the likelihood
  // in this case, ln(L) = -1/2 * (A - B²/C), so we add that on each step
  // reference: https://mc-stan.org/docs/reference-manual/increment-log-prob.html
  target += -0.5*(A - B^2/C);
}
