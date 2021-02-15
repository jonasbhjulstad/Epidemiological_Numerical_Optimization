function Idot = I_ode(I, alpha, beta, N)
    Idot = beta*I*(N-I)-alpha*I;
end
    