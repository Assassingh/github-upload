function plotFeat(FeatStat,FeatNames,num_on_bar)
% FeatStat: nx2 where n is the number of features.  
%     The second dimension is feature number and then score
% FeatNames: Names of all the features
% num_on_bar the number of features to show on the bar graph


%% Sort top feature and plot on bar graph
figure;
FeatStat = sortrows(FeatStat,-2);
barh(FeatStat(num_on_bar:-1:1,2));
FeatNames = FeatNames(FeatStat(:,1));
set(gca,'YTick', 1:num_on_bar,'YTickLabel',FeatNames(num_on_bar:-1:1),'FontSize', 14);
ylim([.5,num_on_bar+.5]);
grid on
xlabel('Feature Criteria Score','FontSize', 18);
ylabel('Features','FontSize', 18);
title(sprintf('Top %d Ranked Features',num_on_bar),'FontSize', 20)