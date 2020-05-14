function [row,dist] = find_closest_points_index_cnst(x_max,y_max,x_pt,y_pt)
%find the closest points from region and constant point
dist_matrix=zeros(length(x_max),1);

for i=1:length(x_max)
    dist_matrix(i,1)=(  (x_max(i)-x_pt).^2 + (y_max(i)-y_pt).^2  ).^0.5;
end

[row,col]=find(dist_matrix==min(min(dist_matrix)));
dist=min(min(dist_matrix));
end

