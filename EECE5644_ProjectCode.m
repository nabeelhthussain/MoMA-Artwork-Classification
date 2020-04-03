%%Program to read-resize-organize color images called from URL's and run
%%PCA on the artwork and then classify by artist nationality.
%%Read table and call urls for images -> RBG matrices

%Wriiten By: Nabeel Hussain, James Tyler, Tzu Chieh Hung, Aydin Wells

full_table = readtable('museum_modern_art_parsed.csv'); % *(NEED TO RECHECK RAW DATA AND CORRECT ERRORS)*
[num,~] = size(full_table);
image_archive = cell(1,num); % Cell holder for all images
NodeNo =  full_table(:,1);
ArtistNo = full_table(:,3);
ArtistNames =  full_table(:,5);
Nationality =  table2array(full_table(:,7));

nat_grps = findgroups(Nationality);
% idx1 = find(nat_grps == 1); %American
idx27 = find(nat_grps == 27); %French
idx42 = find(nat_grps == 42); %Japanese
idx = [idx27(1:end/2);idx42(1:end)];
num = length(idx);


URLlist = string(table2array(full_table(idx,19))); % URLs for all images

for i = 1:num % If error, "Cannot read URL," run loop at error image count and continue
    options = weboptions('Timeout', 30);
    url_img = webread(URLlist(i), options);
    resized_img = imresize(url_img, [300 300]); % Make images same size for PCA
    if sum(size(resized_img)) == 603
        image_archive{i,1} = resized_img; % save uint8 mtx times 3 (RBG) into cell
    end
    if mod(i,50) == 0
        fprintf('Loading Image Number: %d\n', i);
    end
end

image_archive = image_archive(~cellfun('isempty',image_archive));

load chirp.mat;
sound(y);

[num, ~] = size(image_archive);

%% Organize RNG mtx's for PCA (every rand ~2000), run PCA and save PCs

tPCAscore = [];
components = 50; % PCA components to use for downstream analysis

universalcountindex = 0;
universalcount = 1;

full_table = full_table(idx,:);

for groupno = 1:1 % Seperate total data into 5 parts to do PCA in chuncks (due to computational limitation)
    count = 1;
    clear group; clear randartist; clear holder; clear nationalityholder
    for j = 1:num % Edit ending range from new image_archive size
       
            holder = image_archive{j};
            holderR = reshape(holder(:,:,1)',[],1);
            holderB = reshape(holder(:,:,2)',[],1);
            holderG = reshape(holder(:,:,3)',[],1);
            holderRBG = vertcat(vertcat(holderR,holderB),holderG); % 1 long pixal vector for each image
            group(:,count) = holderRBG;
            nationalityholder(count,:) = full_table(j,7);

            count = count + 1;
            universalcount = universalcount + 1;
            
            if mod(universalcount,50) == 0
                fprintf('Loading PCA Number: %d\n', universalcount);
            end        
    end
    universalcountindex(groupno+1,1) = universalcount - 1;
    [coeff,score,latent] = pca(transpose(cast(group,'single'))); % Load PCA -(after transpose): row=artwork,column=PCs
    tPCAscore((universalcountindex(groupno,1)+1):(universalcountindex(groupno+1,1)),:) = score(:,1:components); % Organize PCA data with correct artiwork number
    nationalities((universalcountindex(groupno,1)+1):(universalcountindex(groupno+1,1)),:) = table2array(nationalityholder);

end

%Scatterplot of PCA scores
figure(1);
grid on; axis square equal;
scatter3(tPCAscore(:,1),tPCAscore(:,2),tPCAscore(:,3),'.');
hold on;
imageNum = size(image_archive,1);
imageWidth = 1000;
for i = 1:imageNum
    img = imresize(image_archive{i}, [imageWidth imageWidth]);
    x = tPCAscore(i,1);
    y = tPCAscore(i,2);
    z = tPCAscore(i,3);
    width = imageWidth;
    xImage = [x x+width; x x+width];   % The x data for the image corners
    yImage = [y y; y y];             % The y data for the image corners
    zImage = [z z; z-width z-width];   % The z data for the image corners
    surf(xImage,yImage,zImage,'CData',img,'FaceColor','texturemap');
end
xlabel('PCA1')
ylabel('PCA2')
zlabel('PCA3')

nat_class = findgroups(nationalities);
[val,~] = size(tPCAscore);

%Perform Cross-Validation
cvp = cvpartition(val,'HoldOut',0.10);
idxTrain = training(cvp);
idxTest = test(cvp);
dataTrain = tPCAscore(idxTrain,:);
dataTest = tPCAscore(idxTest,:);

% Just use SVM and Adaboost Classification models for comparison, but we
% tested all of these to determine which was best...

mdlsvm = fitcecoc(dataTrain(:,1:2), nat_class(idxTrain));
% mdlsvm2 = fitcsvm(dataTrain(:,1:2), nat_class(idxTrain));
% mdltree = fitctree(dataTrain(:,1:2), nat_class(idxTrain));
% mdlKnn = fitcknn(dataTrain(:,1:2), nat_class(idxTrain));
% mdlDA = fitcdiscr(dataTrain(:,1:2),nat_class(idxTrain)); 
% t = templateTree('MaxNumSplits',5,'Surrogate','on');
% mdlEn = fitensemble(dataTrain(:,1:2),nat_class(idxTrain),'LogitBoost',100,'Tree');
mdlEn2 = fitensemble(dataTrain(:,1:2),nat_class(idxTrain),'AdaBoostM1',100,'Tree'); 
% mdlEn3 = fitensemble(dataTrain(:,1:2),nat_class(idxTrain),'Bag',100,'Tree','Type','classification'); 

%Generate Confusion Matrix
predictedLabels_svm = predict(mdlsvm, dataTest(:,1:2));
confMat = confusionmat(nat_class(idxTest), predictedLabels_svm)
% predictedLabels_svm2 = predict(mdlsvm2, dataTest(:,1:2));
% confMat = confusionmat(nat_class(idxTest), predictedLabels_svm2)
% predictedLabels_tree = predict(mdltree, dataTest(:,1:2));
% confMat = confusionmat(nat_class(idxTest), predictedLabels_tree)
% predictedLabels_knn = predict(mdlKnn, dataTest(:,1:2));
% confMat = confusionmat(nat_class(idxTest), predictedLabels_knn)
% predictedLabels_lda = predict(mdlDA, dataTest(:,1:2));
% confMat = confusionmat(nat_class(idxTest), predictedLabels_lda)
% predictedLabels_en = predict(mdlEn, dataTest(:,1:2));
% confMat = confusionmat(nat_class(idxTest), predictedLabels_en)
predictedLabels_en2 = predict(mdlEn2, dataTest(:,1:2));
confMat = confusionmat(nat_class(idxTest), predictedLabels_en2)
% predictedLabels_en3 = predict(mdlEn3, dataTest(:,1:2));
% confMat = confusionmat(nat_class(idxTest), predictedLabels_en3)


%Generate ROCs for both models 
tot_op = predictedLabels_svm;
targets = nat_class(idxTest);
th_vals= sort(tot_op);

for i = 1:length(th_vals)
  b_pred = (tot_op>=th_vals(i,1));
  TP = sum(b_pred == 1 & targets == 2);
  FP = sum(b_pred == 1 & targets == 1);
  TN = sum(b_pred == 0 & targets == 1);
  FN = sum(b_pred == 0 & targets == 2);
  sens(i) = TP/(TP+FN);
  spec(i) = TN/(TN+FP);
end

figure(2);
cspec = 1-spec;
cspec = cspec(end:-1:1);
cspec(1,1) = 0;
sens = sens(end:-1:1);
sens(1,1) = 0;
plot(cspec,sens,'k');
xlabel('1 - specificity')
ylabel('sensitivity')

AUC = sum(0.5*(sens(2:end-1)+sens(1:end-2)).*(cspec(2:end-1) - cspec(1:end-2)));
fprintf('\nAUC: %g \n',AUC);
title(['ROC for SVM, AUC = ',num2str(AUC)]);

tot_op = predictedLabels_en2;
targets = nat_class(idxTest);
th_vals= sort(tot_op);

for i = 1:length(th_vals)
  b_pred = (tot_op>=th_vals(i,1));
  TP = sum(b_pred == 1 & targets == 2);
  FP = sum(b_pred == 1 & targets == 1);
  TN = sum(b_pred == 0 & targets == 1);
  FN = sum(b_pred == 0 & targets == 2);
  sens(i) = TP/(TP+FN);
  spec(i) = TN/(TN+FP);
end

figure(3);
cspec = 1-spec;
cspec = cspec(end:-1:1);
cspec(1,1) = 0;
sens = sens(end:-1:1);
sens(1,1) = 0;
plot(cspec,sens,'k');
xlabel('1 - specificity')
ylabel('sensitivity')

AUC = sum(0.5*(sens(2:end-1)+sens(1:end-2)).*(cspec(2:end-1) - cspec(1:end-2)));
fprintf('\nAUC: %g \n',AUC);
title(['ROC for Adaboost, AUC = ',num2str(AUC)]);