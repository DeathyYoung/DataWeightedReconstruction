function [ ] = ranking(incidencePath, edgeWeightPath, initialScorePath, u, savePath)
%incidencePath:		the path of the incidence matrix.
%edgeWeightPath:	the path of the edge's diagonal weight matrix.
%initialScorePath:	the path of the vertice's initial ranking score vector.
%u:					the tradeoff parameter.
%savePath:			the path to save the vertice's final ranking score vector.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Regularization ranking.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
incidenceMatrix = load(incidencePath);
edgeWeightMatrix = load(edgeWeightPath);
initialScoreVector = load(initialScorePath);
alpha = 1 / (1+u);
[v,e] = size(incidenceMatrix);

D_v = zeros(v,v);
for i = 1: v
    tot =0;
    for j = 1:e
        tot = tot + edgeWeightMatrix(j,j)*incidenceMatrix(i,j);
    end
    D_v(i,i) = tot;
end

D_e = zeros(e,e);
for i = 1: e
    D_e(i,i) = sum(incidenceMatrix(:,i));
end

I = eye(v,v);
I_e = eye(e,e);
A = (D_v+I*0.01)^(-1/2) * incidenceMatrix * edgeWeightMatrix * (D_e+I_e*0.01)^(-1/2) * incidenceMatrix' * (D_v+I*0.01)^(-1/2);
rankingScoreVector = inv(I - alpha*A + 0.01*I) * initialScoreVector;
file = fopen(savePath,'w+');
fprintf(file,'%g\r\n',rankingScoreVector);

end