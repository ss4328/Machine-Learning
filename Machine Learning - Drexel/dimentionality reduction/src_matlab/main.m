
%Read in the list of files
D = dir('yalefaces/*');
data_matrix = zeros(154,1600);

flatten = [];

%process files to matrix
for i = 1:length(D)
    if contains(D(i).name, 'subject')
        baseFileName = D(i).name;
        fullFileName = fullfile('yalefaces', D(i).name);
        im = imread(fullFileName);
        sml = imresize(im,[40,40]);
        flatten(end+1,:) = sml(:);   %no preallocating for speed here
    end
end

%preprocessing: substract mean and divide by std deviation
column=0;
clear k row;column;
row = 1;
mean_val=0
for column = 1:1600
    m = mean(flatten(1:154,column));
    s = std(flatten(1:154,column));
    flatten(:,row) = (flatten(:,row) - m) ./s;
    row = row +1;
    
end

mean_value = mean(flatten);
std_deviation = std(flatten);

%now we convert to 2-d PCA matrix
%Reduces the data to 2D using PCA
c = cov(flatten);                       %find covarience matrix
[V, VR]= eig(c);                %find eigenvalues, eigenvectors of the matrix

OVR = VR;                       %replicating the original eigenvectors for reconstructional use

eigenFaces =[];                  %empty array to hold data for eigenfaces

[VR, ind] = sort(diag(VR), 'descend');      %sorts eigenvectors in descending order
pca=zeros(size(flatten,1),2);
pca(:,1) = flatten * V(:,ind(1));
pca(:,2) = flatten * V(:,ind(2));



 
%scatter plot the data; 154 points 
clear i row;column;
kval = 1;
axis1 = pca(:,kval);        %pca is a variable name!
axis2 = pca(:,kval+1);
scatter(axis1,axis2);

%figure out k
topOfMatrix = [];
competencyFactor = 0.95;
max = VR
% max = fliplr(max(VR));        %no need
sumBottom = sum(sum(abs(VR)));

k=35;           %from python code :/    np.cumsum()
for y = 1:size(VR,2)
    val = max(1,y);
    topOfMatrix(:, end+1) = val;
    sumOfTop = sum(topOfMatrix);
    if sumOfTop/sumBottom >= 0.95
        k = y;
        break
    end
end



original = flatten(1,:);             %this is the original data
smImage = reshape(original, [40,40])


v = VideoWriter("animation.avi");
v.FrameRate = 5;
open(v);


 for D = 0:k
     wvals = V(:,(1600-D)+1:1600);
     test_image = flatten(:,:, 1)
     [r c] = size(test_image);
     temp = reshape(test_image',r*c,1);
     temp = double(temp)-mean_value;
     img = imagesc(temp)
     imshow(img);
%      z = flatten(1,:)*wvals;
%      figure;
%      set(gcf, 'colormap' ,gray);
        
%      DImage = image(image,'CDataMapping','scaled');
%      title("image");
%  %     frame = ??
%      writeVideo(v,DImage);
 end

% for D = 0:k
%     
%     W_k = [];
%     for i = 1:D
%         W_k = [W_k V(:,idx(i))]; 
%     end
%     D_k = flatten(1,:);
%     Z_k = D_k * W_k;
%     x_k = Z_k * W_k';
%     im_x_k = reshape(x_k, [40,40]);
%     writeVideo(v,im_x_k);
% end

close(v);