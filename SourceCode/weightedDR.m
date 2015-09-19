function [ ] = weightedDR( X_path, R, U_path, result_path)
%X_path:		the path of sentence's feature matrix.
%R:				the tradeofff parameter.
%U_path:		the path of sentence's weight vector.
%result_path:	the path to save the result.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%weighted data reconstruction revised upon paper: Document Summarization Based on Data Reconstruction.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X = load(X_path);			%sentences' feature matrix.
U = load(U_path);			%sentence's weight vector.
weight = abs(diag(U));		%setnence's diagonal 
[nSent, nSamp] = size(X);

Alpa = abs(rand(nSent, nSent)); 
Alpa_init = Alpa;

Beta = zeros(nSent,1);
Beta_init = Beta;

alpa_min_diff = 1e-4;
beta_min_diff = 1e-6;
alpa_diff = 0;
beta_diff = 0;

beta_flag = 1;
while(beta_flag)
	%calculate the beta based on alpha.
    for i=1:nSent
        Beta(i) = sqrt(sum(Alpa(:,i).^2)/R);
    end
    vectorValid = find(Beta./max(Beta) > 0.1);
    vectorZero = find(Beta./max(Beta) <= 0.1);
    Beta(vectorZero) = 0;
    numValid = sum(Beta./max(Beta) > 0.1);
    vValid = X(vectorValid,:);
    VaVa = vValid * vValid';
    VVa = weight * X * vValid';
    Beta_inverse = inv(diag(Beta(vectorValid)));
    
	%calculate the alpha based on fixed beta.
    alpa_flag = 1;
    while (alpa_flag)
        aValid = Alpa(:,vectorValid);
        AVV = weight * aValid*VaVa;
        AB = aValid*Beta_inverse;
        M = AVV + AB;
        for i=1:nSent;
            v=0;
            for j = 1:numValid;
                v=vectorValid(j);
                if (M(i,j)==0)
                     Alpa(i,j) =0;
                else
                    Alpa(i,v)=Alpa_init(i,v)*VVa(i,j)/M(i,j);
                end
            end
            Alpa(i,vectorZero) = 0;
        end
        
		%judge whether alpha converages.
        alpa_diff = max(max(abs(Alpa_init - Alpa)));
        if (alpa_diff<alpa_min_diff)
            alpa_flag = 0; % Alpa converges;
        else
            Alpa_init = Alpa;
        end
    end % for while (diff_flag)
	
	%judge whether beta converages.
	beta_diff = max(max(abs(Beta_init - Beta)));
    if (beta_diff<beta_min_diff)
        beta_flag = 0; % Beta converges;
    else
        Beta_init = Beta;
    end
end % for while(beta_flag)

C = zeros(1,nSent);
for i= 1:nSent
    [maxTerm, C(1,i)] = max(Beta');
    Beta(C(1,i),1) = -1;
end

file = fopen(result_path,'w+');
fprintf(file,'%g\r\n',C);

end