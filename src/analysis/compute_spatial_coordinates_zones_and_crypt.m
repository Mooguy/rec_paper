%% ===============================
%  Reconstruction + Zone Assignment
%  ===============================

%% Load scRNAseq data
seq_data = importdata('./data/external/CSC_dataset_seurat_raw.data_export_171212.txt');
[dd,tSNE_1,tSNE_2,cluster] = textread('./data/external/CSC_dataset_seurat_tsne_export_171212.txt','%s%f%f%s','headerlines',1);
gene_names_scRNAseq = textread('./data/external/CSC_dataset_seurat_genes_export_171212.txt','%s');

cid=zeros(length(cluster),1);
cid(strcmp(cluster,'enterocyte_1'))=1;
cid(strcmp(cluster,'enterocyte_2'))=2;
cid(strcmp(cluster,'enterocyte_3'))=3;
cid(strcmp(cluster,'enterocyte_4'))=4;
cid(strcmp(cluster,'enterocyte_5'))=5;
cid(strcmp(cluster,'enterocyte_6'))=6;
cid(strcmp(cluster,'transient_amp_1'))=8;
cid(strcmp(cluster,'transient_amp_2'))=9;

scRNAseq_mat = seq_data.data;

% Normalize by total UMIs per cell
scRNAseq_mat_norm = scRNAseq_mat ./ repmat(sum(scRNAseq_mat),size(scRNAseq_mat,1),1);

%% ===============================
%  Handle LCM Data
%  ===============================

load('./data/external/data_LCM.mat')

mat_norm = mat_full ./ repmat(sum(mat_full),size(mat_full,1),1);
Ma = mat_norm(:,1:3:end);
Mb = mat_norm(:,2:3:end);
Mc = mat_norm(:,3:3:end);

Mall(:,:,1)=Ma;
Mall(:,:,2)=Mb;
Mall(:,:,3)=Mc;

mat = mean(Mall,3);

TPM_THRESHOLD = 5e-4;
m = max(mat,[],2);
indin = find(m > TPM_THRESHOLD);

mat = mat(indin,:);
gene_name = gene_name(indin);

%% ===============================
%  Select Landmark Genes
%  ===============================

com=zeros(size(mat,1),1);
mx=zeros(size(mat,1),1);

for i=1:size(mat,1)
    com(i)=sum((1:5).*mat(i,:)/sum(mat(i,:)));
    [~,mx(i)] = max(mat(i,:));
end

THRESH = 1e-3;
ind_low_LCM  = find(m(indin)>THRESH & com<2.5 & mx==1);
ind_high_LCM = find(m(indin)>THRESH & com>3.5 & mx==5);

genes_low  = intersect(gene_names_scRNAseq, gene_name(ind_low_LCM));
genes_high = intersect(gene_names_scRNAseq, gene_name(ind_high_LCM));

% Indices in scRNAseq
ind_low=[];
for i=1:length(genes_low)
    ind_low=[ind_low find(strcmpi(gene_names_scRNAseq,genes_low{i}),1)];
end

ind_high=[];
for i=1:length(genes_high)
    ind_high=[ind_high find(strcmpi(gene_names_scRNAseq,genes_high{i}),1)];
end

%% ===============================
%  Compute scRNA Spatial Coordinate
%  ===============================

vec_low  = sum(scRNAseq_mat_norm(ind_low,:));
vec_high = sum(scRNAseq_mat_norm(ind_high,:));

ind_selected_cells = find(cid<=6 | (cid>=8 & cid<=9));

indicator = vec_high(ind_selected_cells) ./ ...
           (vec_low(ind_selected_cells) + vec_high(ind_selected_cells));

%% ===============================
%  Compute LCM Reference Indicator
%  ===============================

% Find same landmark indices in LCM
ind_low_LCM_final=[];
for i=1:length(genes_low)
    ind_low_LCM_final=[ind_low_LCM_final find(strcmpi(gene_name,genes_low{i}),1)];
end

ind_high_LCM_final=[];
for i=1:length(genes_high)
    ind_high_LCM_final=[ind_high_LCM_final find(strcmpi(gene_name,genes_high{i}),1)];
end

vec_low_LCM  = sum(mat(ind_low_LCM_final,:));
vec_high_LCM = sum(mat(ind_high_LCM_final,:));

indicator_LCM = vec_high_LCM ./ (vec_low_LCM + vec_high_LCM);

%% ===============================
%  Assign Zones
%  ===============================

boundaries = [0 indicator_LCM 1];

zone = zeros(length(indicator),1);

for i=1:length(boundaries)-1
    zone(indicator>=boundaries(i) & ...
         indicator<boundaries(i+1)) = i;
end

% Include right edge
zone(indicator==1) = length(boundaries)-1;

%% ===============================
%  Export Cell-Level Table
%  ===============================

barcodes = seq_data.textdata(1,1:end);
barcodes_selected = barcodes(ind_selected_cells)';
indicator_col = indicator';
zone_col = zone;

T_cells = table(barcodes_selected, indicator_col, zone_col, ...
    'VariableNames', {'Barcode','Spatial_Coordinate','Zone'});

writetable(T_cells,'cell_spatial_coordinates_and_zones.csv');

%% ===============================
%  Export Landmark Genes
%  ===============================

% Combine low and high landmark genes into one cell array
all_landmark_genes = [genes_low; genes_high];

% Write to landmark_genes.txt
fid = fopen('landmark_genes.txt', 'w');
for i = 1:length(all_landmark_genes)
    fprintf(fid, '%s\n', all_landmark_genes{i}); % Write each gene name on a new line
end
fclose(fid);

%% ===============================
%  Export Zone Cutoffs
%  ===============================

Zone_Index = (1:length(boundaries)-1)';
Lower_Bound = boundaries(1:end-1)';
Upper_Bound = boundaries(2:end)';

T_cutoffs = table(Zone_Index,Lower_Bound,Upper_Bound);

writetable(T_cutoffs,'zone_cutoff_values.csv');

disp('Export complete:');
disp('- cell_spatial_coordinates_and_zones.csv');
disp('- zone_cutoff_values.csv');
disp('- landmark_genes.txt');
