
using LinearAlgebra
using CSV, DelimitedFiles
using JuMP, NEOSServer

model = Model() do
    return NEOSServer.Optimizer(; email = "jacksongernesto@hotmail.com", solver = "Ipopt")
end
###########################

#Parameters (Candidate functions)
#f_L(x) = f0 + f1' x + f2' \bar{r}(y(x)) + f3' \bar{r}(-y(x))
#f_L(x) = ([f0 f1' f2' fc2'] + alpha' [e C (D-I) I])*[1; x; \bar{r}(y(x)); \bar{r}(-y(x))]

# f0 = zeros(1);
# f1 = zeros(n);
# f2 = zeros(m);
# fc2 = zeros(m);

f0 = 1;
f1 = [1];
f2 = [1];
fc2 = [1];

f1_row = size(f1,1) 
# f1_col = size(f1,2)
f2_row = size(f2,1) 

n = f1_row #Number of elements of f1 \in \Re^{n}
m = f2_row #Number of elements of f2 \in \Re^{m}

# y - D \bar{r}(y) = Cx + e
e = zeros(m);
C = zeros(m,n);
D = zeros(m,m);

e = ones(m);
C = ones(m,n);
D = zeros(m,m);

#S = {x | g_i(x) \geq 0, i \in {1,...,mc}}
#g_i(x) = (g_i + \beta' [e_{gi} C_{gi} (D_gi - I) I])[1; x; \bar{r}(y_{gi}(x)); \bar{r}(-y_{gi}(x))]
mc = 2; #Number of i elements of S = {x | g_i(x) \geq 0, i \in {1,...,mc}}

g0 = zeros(mc);
g1 = zeros(mc,n);
g2 = zeros(mc,m);
gc2 = zeros(mc,m);

e_g = zeros(mc,m);
C_g = zeros(mc,m,n);
D_g = zeros(mc,m,m);


g0[1] = 1;
g1[1,:] .= 1;
g2[1,:] .= 1;
gc2[1,:] .= 1;

e_g[1,:] .= 1;
C_g[1,:] .= 1;
D_g[1,:] .= 0;


g0[2] = -1;
g1[2,:] .= -1;
g2[2,:] .= -1;
gc2[2,:] .= -1;

e_g[2,:] .= -1;
C_g[2,:] .= -1;
D_g[2,:] .= 0;

#Variables
@variable(model, 0 <= lambda[1:mc])
@variable(model, lambdabeta[1:mc,1:m])
@variable(model, alpha[1:m])
@variable(model, 0 <= gamma)

#Objective
@objective(model, Max, gamma)

@constraints(model, begin
#f_L(x) - \gamma) - \sum \lambda_i g_i(x) \geq 0
  f0 + e'*alpha - gamma  - sum(lambda[i]*g0[i]    + e_g[i,:]'*lambdabeta[i,:] for i in 1:mc)             >= 0
  f1 + C'*alpha          - sum(lambda[i]*g1[i,:]  + C_g[i,:,:]'*lambdabeta[i,:] for i in 1:mc)          .== 0
  f2 + (D - I(m))'*alpha - sum(lambda[i]*g2[i,:]  + (D_g[i,:,:] - I(m))'*lambdabeta[i,:] for i in 1:mc) .>= 0
  fc2 + alpha            - sum(lambda[i]*gc2[i,:] + lambdabeta[i,:] for i in 1:mc)                      .>= 0
end )

#Execution
print(model)
optimize!(model)
print(solution_summary(model))

#Save variable values
alpha = value.(alpha)
beta = value.(lambdabeta)./value.(lambda)
lambda = value.(lambda)
gamma = value.(gamma)

#Export solutions
writedlm("alpha.csv", alpha,',')
writedlm("beta.csv", beta, ',')
writedlm("lambda.csv", lambda, ',')
writedlm("gamma.csv",gamma , ',')

open("results.txt", "a") do io
  print(io, solution_summary(model; verbose = true))
end