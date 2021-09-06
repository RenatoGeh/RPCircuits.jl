function simple_circuit()::Node
  x, y, z, w = Categorical(1, [0.6, 0.4]),
               Categorical(1, [0.1, 0.9]),
               Categorical(2, [0.3, 0.7]),
               Categorical(2, [0.8, 0.2])
  return 0.2*(x*z) + 0.5*(x*w) + 0.3*(y*w)
end

function hmm()::Node
  a, b, c, d, e, f = Categorical(1, [0.3, 0.7]),
                     Categorical(1, [0.8, 0.2]),
                     Categorical(2, [0.4, 0.6]),
                     Categorical(2, [0.25, 0.75]),
                     Categorical(3, [0.9, 0.1]),
                     Categorical(3, [0.42, 0.58])
  p_1, p_2 = c*(0.6*e + 0.4*f), (0.4*e + 0.6*f)*d
  s_1, s_2 = 0.5*p_1 + 0.5*p_2, 0.2*p_1 + 0.8*p_2
  return 0.3*(a*s_1) + 0.7*(s_2*b)
end

function selspn()::Node
  i_1, i_2 = Indicator(1, 1.0), Indicator(1, 2.0)
  a, b, c, d = Categorical(2, [0.3, 0.7]),
               Categorical(3, [0.4, 0.6]),
               Categorical(2, [0.8, 0.2]),
               Categorical(3, [0.9, 0.1])
  return 0.4*(i_2*a*b)+0.6*(c*d*i_1)
end

function psdd()::Node
  a_1, a_2 = Indicator(1, 1.0), Indicator(1, 2.0)
  b_1, b_2 = Indicator(2, 1.0), Indicator(2, 2.0)
  c_1, c_2 = Indicator(3, 1.0), Indicator(3, 2.0)
  d_1, d_2 = Indicator(4, 1.0), Indicator(4, 2.0)
  w_1 = exp.([-1.6094379124341003, -1.2039728043259361, -0.916290731874155, -2.3025850929940455])
  w_2 = exp.([-2.3025850929940455, -2.3025850929940455, -2.3025850929940455, -0.35667494393873245])
  return Sum([c_2*d_2, c_2*d_1, c_1*d_2, c_1*d_1], w_1)*Sum([a_2*b_2, a_2*b_1, a_1*b_2, a_1*b_1], w_2)
end

function gaussian_circuit()::Node
  x, y, z, w = Gaussian(1, 2, 18), Gaussian(1, 11, 8), Gaussian(2, 3, 10), Gaussian(2, -4, 7)
  return 0.2*(x*z)+0.45*(x*w)+0.35*(y*w)
end

