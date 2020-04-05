function myKMeans = newFunc(X,Y,k)
    %restraining the value of k to 7
    if(k>7)
        k=7;
    end
    disp('myKMeans func working with k as'+ num2str(k));

    %defining colors
    colors = ['r','g','b','y','c','m','k'];
    videoFrames = [];

    %rows,cols for later usage
    [rowsX,colsX] = size(X);

    if(colsX>3)
        coeff = pca(X);
        reduceMat = coeff(:,1:3);   %reduce to 3 pca features only
        X = X*reduceMat;
    end

    %calculating means
    %means = [];
    means = zeros(size(X,1), size(X,2));
    disp('Calculating Means');
    indexes = randperm(rowsX,k); %k random numbers
    for j = 1:length(indexes)
        %means = [means;X(indexes(j),:)];
        means(j,:) = X(indexes(j),:);
    end

    maxChange = 2^-23;
    change = 1.0;

    i=0;
    videoIteration = 1;
    
    while 3>0
        if change>maxChange
           disp(num2str(change));
           i=i+1;
           dist =[];
           purity =0;

           %this loop calculates the new distance
           j=1;
           while j<=k
               calc =[];
               for n =1:rowsX
                   calc = [calc; norm(means(j,:)-X(n,:))];
               end
               dist = [dist calc];
               j=j+1;
           end

           disp('Creating Clusters');
            %this loop creates clusters
            Clust(1:rowsX,1:3,1:k) = 0;
            for i = 1:rowsX 
                [v,index2] = min(dist(i,:));
                Clust(i,:,index2) = X(i,:);
            end

            %HERE

            %copy each cluster to memory
            j=1;
            while j<=k
                memory = [Clust(:,:,j) Y];
                memory2 = [];
                % For loop to remove zero vectors or combine two matrices
                for n = 1:rowsX
                    if memory(n,1) == 0 && memory(n,2) == 0 && memory(n,3) == 0
                        continue
                    else
                        memory2 = [memory2;memory(n,:)];
                    end
                end
                ones = sum(memory2(:,4) == 1);
                minusOnes = sum(memory2(:,4) == -1);
                purity = purity + max(ones,minusOnes);
                j=j+1;
            end

            %purity (which will be printed)
            purity = purity/rowsX;

            imageName = ['Count ', num2str(videoIteration),', Purity' ,num2str(purity)];
            clf;

            % Plot every cluster, plot the values in that cluster    
            j=1;
             if(size(X,2)==3)
                 while j<=k
                    color = colors(j);
                    scatter3(Clust(:,1,j),Clust(:,2,j),Clust(:,3,j),36,color,'x');      %Plot Cluster i
                    hold all;
                    scatter3(means(j,1),means(j,2),means(j,3),75,'MarkerEdgeColor','k','MarkerFaceColor',color);
                    title(imageName);
                    j=j+1;
                end
             end
            
            %figures dir contains the plots 
            addpath('figures');

            name = ['figures/figure',num2str(videoIteration),'.jpg']; 
            videoFrames = [videoFrames;string(name)];
            saveas(figure(i),name)

            %Calculate new means
            newMeans = means;       
            means = [];
            j=1;
            while j<=k
                memory = Clust(:,:,j);
                memory = memory(any(memory,2),:);
                means = [means;mean(memory)];
                j=j+1;
            end

            change = 0;
            j=1;
            while j<=k
                Di = (means(j,1)-newMeans(j,1))+(means(j,2)-newMeans(j,2))+(means(j,3)-newMeans(j,3));
                change = change + Di;
                j=j+1;
            end
            videoIteration = videoIteration+1;
        else
            break;
        end
    end         %og loop ends
    
    disp('Generating video');
    videoName = ['K_',num2str(k),'_F_all'];
    videoObj = VideoWriter(videoName, 'MPEG-4');
    videoObj.FrameRate = 5;
    open(videoObj);
    [img,Clust] = size(videoFrames);
    a=1;
    while a<=img
      I = imread(char(videoFrames(a)));
      writeVideo(videoObj,I);
      a=a+1;
    end
    close(videoObj);
end