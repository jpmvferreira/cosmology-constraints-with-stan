functions {
  // return E(z)
  real E(real z, array[] real theta) {
    real Ob = theta[2];
    real Oc = theta[3];
    real Or = theta[4];
    real OL = theta[5];

    return ((Ob + Oc)*(1.0+z)^3 + Or*(1.0+z)^4 + OL)^0.5;
  }

  // return auxiliary function f(z)
  real f(real z, array[] real theta) {
   real h = theta[1];
   real Ob = theta[2];

   real Tcmb = 2.7255;  // Zhai2018
   real Rb = 31500*Ob*(h^2)*((2.7/Tcmb)^4);

   return (3*(1+Rb/(1+z)))^0.5;
  }

  // return v'(z) ≡ 1/E(z)
  vector dvdz(real z, vector y, array[] real theta) {
    vector[1] yderiv = [1.0/E(z, theta)]';
    return yderiv;
  }

  // return 1/[E(z)*f(z)]
  real integrand(real x, real xc, array[] real theta, array[] real x_r, array[] int x_i) {
    return 1/(E(x, theta)*f(x, theta));
  }
}

data {
  real la_exp;
  real la_error;
  real R_exp;
  real R_error;
  real wb_exp;
  real wb_error;
}

transformed data {
  array[0] real x_r;  // create null data values to match the signature required by integrate_1d
  array[0] int x_i;   // docs: https://mc-stan.org/docs/functions-reference/functions-1d-integrator.html

  // inverse of the covariance matrix
  // computed based on the data from arXiv:2302.04807, which uses Planck 2018 TT+lowE
  matrix[3,3] Cinv;
  Cinv[1,1] = 70.41272413  ; Cinv[1,2] = -472.33176518  ; Cinv[1,3] = 4150.71131541;
  Cinv[2,1] = -472.33176518; Cinv[2,2] = 31027.46200904 ; Cinv[2,3] = 477813.59257032;
  Cinv[3,1] = 4150.71131541; Cinv[3,2] = 477813.59257032; Cinv[3,3] = 30083780.34236922;
}

parameters {
  // lower bound for the Hubble constant is 0
  real<lower=0> h;

  // no lower bound because we expect small values and we don't want to mess up the chains
  // also provides extra statistical freedom
  real Ob;
  real Oc;
}

transformed parameters {
  // Ω_r is a derived parameter from ω_r = Ω_r * h²
  // no lower bound because it's expected to be close to 0
  real wr = 4.15*10^(-5);
  real Or = wr/h^2;

  // conservation laws to obtain Ω_Λ
  real OL = 1 - Oc - Ob - Or;

  // (pseudo) parameter array
  array[5] real theta = {h, Ob, Oc, Or, OL};

  // compute zstar
  real g1 = (0.0783*(Ob*h^2)^(-0.238)) / (1 + 39.5*(Ob*h^2)^0.763);
  real g2 = 0.560 / (1 + 21.1*(Ob*h^2)^1.81);
  real zstar = 1048 * (1 + 0.00124*(Ob*h^2)^(-0.738)) * (1+g1*((Ob + Oc)*h^2)^g2);

  // compute v ≡ v(zstar) ≡ the integral of 1/E(x) from 0 to zstar
  real z0 = 0;          // initial conditions: v(0) = 0
  vector[1] v0 = [0]';  // .
  array[1] vector[1] sol_v = ode_rk45(dvdz, v0, z0, {zstar}, theta);
  real v = sol_v[1][1];

  // compute the integral of 1/(E(x)*f(x)) from zstar to ∞
  real g = integrate_1d(integrand, zstar, positive_infinity(), theta, x_r, x_i);

  // shift parameters
  real la = pi()*v/g;
  real R = sqrt(Ob + Oc)*v;
  real wb = Ob*h^2;

  // difference vector between the model predictions and the observations
  vector[3] x = [la - la_exp, R - R_exp, wb - wb_exp]';
}

model {
  // priors
  h ~ normal(0.7, 3);
  Ob ~ normal(0.05, 3);
  Oc ~ normal(0.25, 3);

  // increase log likelihood
  target += -x'*Cinv*x;
}
