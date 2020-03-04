function bbxs_table_new = get_bbxs_csv(filepath,W,H,x,y)
%loading the table
%This can be used to get info about cropped area by using offset
% contents are ID< centroid_x, centroid_y, xmin, ymin, xmax, ymax, NeuN,
% S100, Olig2, IBA1, RECA1

bbxs_table=readtable(filepath);
bbxs_table.xmax=bbxs_table.xmax-x;
bbxs_table.xmin=bbxs_table.xmin-x;
bbxs_table.centroid_x=bbxs_table.centroid_x-x;

bbxs_table.ymax=bbxs_table.ymax-y;
bbxs_table.ymin=bbxs_table.ymin-y;
bbxs_table.centroid_y=bbxs_table.centroid_y-y;

bbxs_table_new=bbxs_table(bbxs_table.xmin>=1,:);
bbxs_table_new=bbxs_table_new(bbxs_table_new.ymin>=1,:);
bbxs_table_new=bbxs_table_new(bbxs_table_new.xmax<=W,:);
bbxs_table_new=bbxs_table_new(bbxs_table_new.ymax<=H,:);

end

