function [ ] = semiPLSA(prior_path, data_path, save_path)
%prior_path:	the path of scientific paper's abstract's feature matrix.
%data_path:		the path of citation sentence's feature matrix.
%sava_path:		the path to save the result, that is each sentence's cluster assignment.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Based on paper: Opinion Integration Through Semi-supervised Topic Modeling.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prior = load(prior_path);	%prior:	abstract's feature matrix
data = load(data_path);		%data:	citation sentence's feature matrix
gamma = 0.5;				%the decay parameter
threshold = 0.001;			%the condition of iteration termination.
Lprev = -inf;

%normalize the prior probability.
[priorDocs,wordSize] = size(prior);
normalizedPrior = zeros(priorDocs,wordSize);
for i = 1: priorDocs
    if sum(prior(i,:)) == 0
        normalizedPrior(i,:) = 0;
    else
        normalizedPrior(i,:) = prior(i,:) / sum(prior(i,:));
    end
end

[docNum,~] = size(data);
extraTopics = round((docNum - priorDocs*1)/3);	%extra topic size, empirically set up.
topicNum = priorDocs + extraTopics;
theta = zeros(topicNum,wordSize);	%the topic-word distribution.

pai = zeros(docNum,topicNum);	%the document-topic distribution.

Z_d_w_j = zeros(docNum,wordSize,topicNum);	%the probability of assigning topic j to word w in sentence d.
for i = 1: docNum
    Z_d_w_j(i,:,:) = rand(wordSize,topicNum);	%randomly initialize.
end

phi = zeros(topicNum);	%the confidience parameter for prior, empirically set up.
for i = 1:priorDocs
    phi(i) = 50;
end
for i = priorDocs+1 : topicNum
    phi(i) = 0;
end

iterations = 0;
while true
    iterations = iterations + 1;
	%update the document-topic distribution.
	for i = 1:docNum
		for j = 1:topicNum
			pai(i,j) = dot(data(i,:),Z_d_w_j(i,:,j));
		end
	end
    if sum(pai(i,:)) == 0
        pai(i,:) = 0;
    else
        pai(i,:) = pai(i,:)/sum(pai(i,:));
    end
	
	%calculate the confidence parameter.
	for i = 1: priorDocs
		tempMatrix = data.*Z_d_w_j(:,:,i);
		tempVal = sum(tempMatrix(:));
        if (phi(i) + tempVal) == 0
            u = 0;
        else
            u = phi(i) / (phi(i) + tempVal);
        end
		if(u > gamma)
			phi(i) = phi(i)*0.9;
		end
	end
	
	%update the topic-word distribution.
	for i = 1: priorDocs
		for j = 1: wordSize
			tempVal_1 = dot(data(:,j),Z_d_w_j(:,j,i));
			tempMatrix = data.*Z_d_w_j(:,:,i);
			tempVal_2 = sum(tempMatrix(:));
            if (tempVal_2 + phi(i)) == 0
                theta(i,j) = 0;
            else
                theta(i,j) = (tempVal_1 + normalizedPrior(i,j)*phi(i)) / (tempVal_2+phi(i));
            end
		end
	end
	for i = priorDocs+1 : topicNum
		for j = 1: wordSize
			tempVal_1 = dot(data(:,j),Z_d_w_j(:,j,i));
			tempMatrix = data.*Z_d_w_j(:,:,i);
			tempVal_2 = sum(tempMatrix(:));
            if tempVal_2 == 0
                theta(i,j) = 0;
            else
                theta(i,j) = tempVal_1 / tempVal_2;
            end
		end
	end

	%calculate the log-likelihood, and judge whether convengence.
	tempMatrix = data.*log(pai*theta + 0.00001);
	L = sum(tempMatrix(:));
	if (L-Lprev < threshold) & (iterations > 200) 
		break;  
	end
	Lprev = L;

	%calculate the topic assignment probability.
	for i = 1: docNum
		for j = 1: wordSize
			tempVal = dot(pai(i,:),theta(:,j));
			for l = 1: topicNum
                if tempVal == 0
                    Z_d_w_j(i,j,l) = 0;
                else
                    Z_d_w_j(i,j,l) = (pai(i,l)*theta(l,j)) / tempVal;
                end
			end
		end
	end

end	%end of while loop.

M = data*theta';	%clustering.
[max_a,index] = max(M,[],2);
file = fopen(save_path,'w+');
fprintf(file,'%g\r\n',index);
end